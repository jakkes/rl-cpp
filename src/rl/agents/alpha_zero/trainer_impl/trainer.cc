#include "trainer.h"

#include <c10/cuda/CUDAStream.h>
#include <rl/torchutils/torchutils.h>

#include "helpers.h"


using namespace torch::indexing;


namespace
{
    class InferenceUnit : public rl::torchutils::ExecutionUnit
    {
        public:
            InferenceUnit(
                int max_batchsize,
                torch::Device device,
                std::shared_ptr<modules::Base> module,
                bool use_cuda_graph
            ) : rl::torchutils::ExecutionUnit(max_batchsize, device, use_cuda_graph), module{module}
            {}
        
        private:
            std::shared_ptr<modules::Base> module;

        private:
            rl::torchutils::ExecutionUnitOutput forward(const std::vector<torch::Tensor> &inputs)
            {
                torch::NoGradGuard no_grad_guard{};
                auto module_output = module->forward(inputs[0]);
                auto policy_output = module_output->policy().get_probabilities();
                auto value_output = module_output->value_estimates();

                rl::torchutils::ExecutionUnitOutput out{2, 0};
                out.tensors[0] = policy_output;
                out.tensors[1] = value_output;

                return out;
            }
    };

    class TrainingUnit : public rl::torchutils::ExecutionUnit
    {
        public:
            TrainingUnit(
                int max_batchsize,
                torch::Device device,
                std::shared_ptr<modules::Base> module,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                bool use_cuda_graph
            ) : rl::torchutils::ExecutionUnit(max_batchsize, device, use_cuda_graph), module{module}, optimizer{optimizer}
            {
                if (use_cuda_graph) {
                    throw std::runtime_error{"CUDAGraph for training not yet supported."};
                }
            }

        private:
            std::shared_ptr<modules::Base> module;
            std::shared_ptr<torch::optim::Optimizer> optimizer;

        private:
            rl::torchutils::ExecutionUnitOutput forward(const std::vector<torch::Tensor> &inputs)
            {
                auto &states = inputs[0];
                auto &posteriors = inputs[1];
                auto &rewards = inputs[2];
                auto module_output = module->forward(states);
                auto policy_loss = module_output->policy_loss(posteriors).mean();
                auto value_loss = module_output->value_loss(rewards).mean();
                auto loss = policy_loss + value_loss;

                optimizer->zero_grad();
                loss.backward();

                auto gradient_norm = rl::torchutils::compute_gradient_norm(optimizer);

                optimizer->step();

                rl::torchutils::ExecutionUnitOutput out{0, 3};
                out.scalars[0] = policy_loss;
                out.scalars[1] = value_loss;
                out.scalars[2] = gradient_norm;

                return out;
            }
    };
}

namespace trainer_impl
{
    Trainer::Trainer(
        std::shared_ptr<rl::simulators::Base> simulator,
        std::shared_ptr<modules::Base> module,
        std::shared_ptr<rl::buffers::samplers::Uniform<rl::buffers::Tensor>> sampler,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        std::shared_ptr<std::mutex> optimizer_step_mtx,
        const TrainerOptions &options
    )
    :   simulator{simulator}, module{module}, sampler{sampler},
        optimizer{optimizer}, optimizer_step_mtx{optimizer_step_mtx}, options{options}
    {
        setup_inference_unit();
        setup_training_unit();
    }

    void Trainer::start() {
        running = true;
        working_thread = std::thread(&Trainer::worker, this);
    }

    void Trainer::stop() {
        running = false;
        if (working_thread.joinable()) {
            working_thread.join();
        }
    }

    void Trainer::worker()
    {
        torch::MultiStreamGuard stream_guard{get_cuda_streams()};

        while (running && sampler->buffer_size() < options.min_replay_size) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        while (running) {
            step();
        }
    }

    torch::Tensor Trainer::get_target_policy(const torch::Tensor &states, const torch::Tensor &masks)
    {
        torch::NoGradGuard no_grad_guard{};

        FastMCTSExecutor mcts_executor{
            states,
            masks,
            inference_fn_var,
            simulator,
            options.mcts_options
        };

        mcts_executor.run();
        auto visit_counts = mcts_executor.current_visit_counts();
        auto temperature = options.temperature_control->get();
        auto probabilities = visit_counts.pow(temperature);
        probabilities /= probabilities.sum(1, true);

        return probabilities;
    }

    void Trainer::step()
    {
        auto sample_storage = sampler->sample(options.batchsize);
        auto &sample{*sample_storage};
        auto &states = sample[0];
        auto &masks = sample[1];
        auto &rewards = sample[2];

        auto posteriors = get_target_policy(states, masks);

        std::unique_lock optimizer_step_guard{*optimizer_step_mtx};
        auto training_outputs = training_unit->operator()({states.to(options.module_device), posteriors.to(options.module_device), rewards.to(options.module_device)});
        optimizer_step_guard.unlock();

        if (options.logger)
        {
            options.logger->log_scalar("AlphaZero/Policy loss", training_outputs.scalars[0].item().toFloat());
            options.logger->log_scalar("AlphaZero/Value loss", training_outputs.scalars[1].item().toFloat());
            options.logger->log_scalar("AlphaZero/Gradient norm", training_outputs.scalars[2].item().toFloat());
            options.logger->log_frequency("AlphaZero/Training rate", 1);
        }
    }

    void Trainer::setup_inference_unit()
    {
        inference_unit = std::make_unique<InferenceUnit>(
            options.batchsize,
            options.module_device,
            module,
            options.enable_cuda_graph_inference
        );
        inference_unit->operator()({simulator->reset(options.batchsize).states.to(options.module_device)});
    }

    FastMCTSInferenceResult Trainer::inference_fn(const torch::Tensor &states) {
        auto outputs = inference_unit->operator()({states});
        return FastMCTSInferenceResult{
            outputs.tensors[0],
            outputs.tensors[1]
        };
    }

    void Trainer::setup_training_unit()
    {
        training_unit = std::make_unique<TrainingUnit>(
            options.batchsize,
            options.module_device,
            module,
            optimizer,
            options.enable_cuda_graph_training
        );
        auto states = simulator->reset(options.batchsize);
        auto masks = states.action_constraints->as_type<rl::policies::constraints::CategoricalMask>().mask();
        auto posteriors = torch::ones(masks.sizes());
        auto rewards = torch::randn({options.batchsize});

        training_unit->operator()({states.states.to(options.module_device), posteriors.to(options.module_device), rewards.to(options.module_device)});
    }
}
