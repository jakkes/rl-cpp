#include "trainer.h"

#include <c10/cuda/CUDAStream.h>
#include <rl/torchutils/torchutils.h>

#include "helpers.h"


using namespace torch::indexing;


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
        auto inference_output = inference_fn(states.to(options.module_device));
        auto priors = inference_output.policies.get_probabilities().to(torch::kCPU);
        auto values = inference_output.values.to(torch::kCPU);

        std::vector<std::shared_ptr<MCTSNode>> nodes{}; nodes.reserve(options.batchsize);
        for (int i = 0; i < options.batchsize; i++) {
            nodes.push_back(
                std::make_shared<MCTSNode>(
                    states.index({i}),
                    masks.index({i}),
                    priors.index({i}),
                    values.index({i}).item().toFloat()
                )
            );
        }

        mcts(&nodes, inference_fn_var, simulator, options.mcts_options);
        auto policy = mcts_nodes_to_policy(nodes, options.temperature_control->get());
        return policy.get_probabilities();
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

    MCTSInferenceResult Trainer::inference_fn(const torch::Tensor &states) {
        auto outputs = inference_unit->operator()({states});
        return MCTSInferenceResult{
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
