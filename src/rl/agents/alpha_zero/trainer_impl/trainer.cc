#include "trainer.h"

#include <c10/cuda/CUDAStream.h>
#include <rl/torchutils.h>

#include "helpers.h"


using namespace torch::indexing;

namespace trainer_impl
{
    Trainer::Trainer(
        std::shared_ptr<rl::simulators::Base> simulator,
        std::shared_ptr<modules::Base> module,
        std::shared_ptr<thread_safe::Queue<SelfPlayEpisode>> episode_queue,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        std::shared_ptr<std::mutex> optimizer_step_mtx,
        const TrainerOptions &options
    )
    :   simulator{simulator}, module{module}, episode_queue{episode_queue},
        optimizer{optimizer}, optimizer_step_mtx{optimizer_step_mtx}, options{options}
    {
        init_buffer();
        inference_fn_setup();
    }

    void Trainer::start() {
        running = true;
        working_thread = std::thread(&Trainer::worker, this);
        queue_consuming_thread = std::thread(&Trainer::queue_consumer, this);
    }

    void Trainer::stop() {
        running = false;
        if (working_thread.joinable()) {
            working_thread.join();
        }
        if (queue_consuming_thread.joinable()) {
            queue_consuming_thread.join();
        }
    }

    void Trainer::init_buffer()
    {
        auto states = simulator->reset(1);
        auto state = states.states.squeeze(0);
        auto mask = get_mask(*states.action_constraints).squeeze(0);
        
        buffer = std::make_shared<rl::buffers::Tensor>(
            options.replay_size,
            std::vector{
                state.sizes().vec(),
                mask.sizes().vec(),
                std::vector<int64_t>{}
            },
            std::vector{
                state.options(),
                mask.options(),
                torch::TensorOptions{}.dtype(torch::kFloat32).device(state.device())
            }
        );

        sampler = std::make_unique<rl::buffers::samplers::Uniform<rl::buffers::Tensor>>(buffer);
    }

    void Trainer::worker()
    {
        torch::StreamGuard stream_guard{cuda_stream};

        while (running && buffer->size() < options.min_replay_size) {
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

        mcts(&nodes, inference_fn, simulator, options.mcts_options);
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

        auto posteriors = get_target_policy(states, masks).to(options.module_device);

        std::unique_lock optimizer_step_guard{*optimizer_step_mtx};
        auto module_output = module->forward(states.to(options.module_device));
        auto policy_loss = module_output->policy_loss(posteriors).mean();
        auto value_loss = module_output->value_loss(rewards.to(options.module_device)).mean();
        auto loss = policy_loss + value_loss;

        optimizer->zero_grad();
        loss.backward();

        auto gradient_norm = rl::torchutils::compute_gradient_norm(optimizer).item().toFloat();
        if (gradient_norm > options.gradient_norm) {
            rl::torchutils::scale_gradients(optimizer, options.gradient_norm / gradient_norm);
        }

        optimizer->step();
        optimizer_step_guard.unlock();

        if (options.logger)
        {
            options.logger->log_scalar("AlphaZero/Value loss", value_loss.item().toFloat());
            options.logger->log_scalar("AlphaZero/Policy loss", policy_loss.item().toFloat());
            options.logger->log_frequency("AlphaZero/Training rate", 1);
        }
    }

    void Trainer::queue_consumer()
    {
        while (running)
        {
            auto episode_ptr = episode_queue->dequeue(std::chrono::seconds(5));
            if (!episode_ptr) {
                continue;
            }

            auto episode = *episode_ptr;
            buffer->add({episode.states, episode.masks, episode.collected_rewards});
        }
    }

    void Trainer::inference_fn_setup()
    {
        if (options.module_device.is_cuda()) {
            cuda_graph_inference_setup();
            inference_fn = std::bind(&Trainer::cuda_graph_inference_fn, this, std::placeholders::_1);
        }
        else {
            inference_fn = std::bind(&Trainer::cpu_inference_fn, this, std::placeholders::_1);
        }
    }

    void Trainer::cuda_graph_inference_setup()
    {
        torch::StreamGuard stream_guard{cuda_stream};
        torch::NoGradGuard no_grad_guard{};

        inference_graph = std::make_unique<at::cuda::CUDAGraph>();
        inference_input = simulator->reset(options.batchsize).states.to(options.module_device);
        auto intermediate_output_1 = module->forward(inference_input);
        inference_policy_output = intermediate_output_1->policy().get_probabilities();
        inference_value_output = intermediate_output_1->value_estimates();

        inference_graph->capture_begin();
        auto intermediate_output_2 = module->forward(inference_input);
        inference_policy_output = intermediate_output_2->policy().get_probabilities();
        inference_value_output = intermediate_output_2->value_estimates();
        inference_graph->capture_end();
    }

    MCTSInferenceResult Trainer::cuda_graph_inference_fn(const torch::Tensor &states)
    {
        auto N = states.size(0);
        inference_input.index_put_({Slice(None, N)}, states);
        inference_graph->replay();
        return MCTSInferenceResult{
            inference_policy_output.index({Slice(None, N)}).clone(),
            inference_value_output.index({Slice(None, N)}).clone()
        };
    }
}
