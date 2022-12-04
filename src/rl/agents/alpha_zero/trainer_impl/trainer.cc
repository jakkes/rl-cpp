#include "trainer.h"

#include <rl/torchutils.h>

#include "helpers.h"


namespace trainer_impl
{
    Trainer::Trainer(
        std::shared_ptr<rl::simulators::Base> simulator,
        std::shared_ptr<modules::Base> module,
        std::shared_ptr<thread_safe::Queue<SelfPlayEpisode>> episode_queue,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        const TrainerOptions &options
    )
    :   simulator{simulator}, module{module}, episode_queue{episode_queue},
        optimizer{optimizer}, options{options}
    {
        init_buffer();
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
                state.options()
            }
        );

        sampler = std::make_unique<rl::buffers::samplers::Uniform<rl::buffers::Tensor>>(buffer);
    }

    void Trainer::worker()
    {
        while (running && buffer->size() < options.min_replay_size) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        while (running) {
            step();
        }
    }

    void Trainer::step()
    {
        auto sample_storage = sampler->sample(options.batchsize);
        auto &sample{*sample_storage};
        auto &states = sample[0];
        auto &masks = sample[1];
        auto &rewards = sample[2];

        auto module_output = module->forward(states);

        auto priors = module_output->policy().get_probabilities().detach();
        auto values = module_output->value_estimates().detach();

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

        mcts(&nodes, module, simulator, options.mcts_options);
        auto policy = mcts_nodes_to_policy(nodes, masks, options.temperature);
        auto posteriors = policy.get_probabilities();

        auto policy_loss = (posteriors * module_output->policy().get_probabilities().log()).sum(1).mean();
        auto value_loss = module_output->value_loss(rewards).mean();
        auto loss = policy_loss + value_loss;

        optimizer->zero_grad();
        loss.backward();

        auto gradient_norm = rl::torchutils::compute_gradient_norm(optimizer).item().toFloat();
        if (gradient_norm > options.gradient_norm) {
            rl::torchutils::scale_gradients(optimizer, options.gradient_norm / gradient_norm);
        }

        optimizer->step();

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
        }
    }
}
