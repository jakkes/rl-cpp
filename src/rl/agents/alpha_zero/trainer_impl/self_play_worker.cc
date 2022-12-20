#include "self_play_worker.h"

#include <rl/utils/reward/backpropagate.h>

#include "helpers.h"


using namespace torch::indexing;

namespace trainer_impl
{
    SelfPlayWorker::SelfPlayWorker(
        std::shared_ptr<rl::simulators::Base> simulator,
        std::shared_ptr<modules::Base> module,
        std::shared_ptr<thread_safe::Queue<SelfPlayEpisode>> episode_queue,
        const SelfPlayWorkerOptions &options
    ) : simulator{simulator}, module{module}, episode_queue{episode_queue}, options{options}
    {
        batchvec = torch::arange(options.batchsize);
    }

    void SelfPlayWorker::start()
    {
        running = true;
        working_thread = std::thread(&SelfPlayWorker::worker, this);
    }

    void SelfPlayWorker::stop()
    {
        running = false;
        if (working_thread.joinable()) {
            working_thread.join();
        }
    }

    void SelfPlayWorker::reset_histories(const torch::Tensor &terminal_mask)
    {
        auto batchsize = terminal_mask.sum().item().toLong();
        auto terminal_indices = torch::arange(options.batchsize).index({terminal_mask});

        auto initial_states = simulator->reset(batchsize);
        auto states = initial_states.states;
        auto masks = get_mask(*initial_states.action_constraints);

        state_history.index_put_({terminal_indices, 0}, states);
        state_history.index_put_({terminal_indices, Slice(1, None)}, 0.0f);
        mask_history.index_put_({terminal_indices, 0}, masks);
        mask_history.index_put_({terminal_indices, Slice(1, None)}, false);
        reward_history.index_put_({terminal_mask}, torch::zeros_like(reward_history.index({terminal_mask})));
        steps.index_put_({terminal_mask}, 0);
    }

    void SelfPlayWorker::reset_mcts_nodes(const torch::Tensor &terminal_mask)
    {
        auto terminal_mask_accessor{terminal_mask.accessor<bool, 1>()};

        auto states = this->state_history.index({terminal_mask}).index({Slice(), 0});
        auto masks = this->mask_history.index({terminal_mask}).index({Slice(), 0});
        auto module_output = module->forward(states);
        auto priors = module_output->policy().get_probabilities();
        auto values = module_output->value_estimates();
        
        int j{-1};
        for (int i = 0; i < options.batchsize; i++)
        {
            if (!terminal_mask_accessor[i]) {
                continue;
            }
            j++;

            mcts_nodes[i] = std::make_shared<MCTSNode>(
                states.index({j}),
                masks.index({j}),
                priors.index({j}),
                values.index({j}).item().toFloat()
            );
        }

        if (options.logger) {
            options.logger->log_scalar("AlphaZero/Start value", values.mean().item().toFloat());
            options.logger->log_scalar("AlphaZero/Start entropy", module_output->policy().entropy().mean().item().toFloat());
        }
    }

    void SelfPlayWorker::set_initial_state()
    {
        auto initial_states = simulator->reset(options.batchsize);
        auto states = initial_states.states;
        auto masks = get_mask(*initial_states.action_constraints);

        mcts_nodes.resize(options.batchsize);

        state_history = states.unsqueeze(1);
        mask_history = masks.unsqueeze(1);

        for (int i = 0; i < options.max_episode_length - 1; i++)
        {
            state_history = torch::concat(
                {
                    state_history,
                    torch::zeros_like(states).unsqueeze(1)
                },
                1
            );

            mask_history = torch::concat(
                {
                    mask_history,
                    torch::zeros_like(masks).unsqueeze(1)
                },
                1
            );
        }

        reward_history = torch::zeros({options.batchsize, options.max_episode_length});
        steps = torch::zeros({options.batchsize}, torch::TensorOptions{}.dtype(torch::kLong));

        reset_mcts_nodes(torch::ones({options.batchsize}, torch::TensorOptions{}.dtype(torch::kBool)));
    }

    torch::Tensor SelfPlayWorker::step_mcts_nodes(const torch::Tensor &actions)
    {
        auto terminals = torch::zeros({options.batchsize}, torch::TensorOptions{}.dtype(torch::kBool));

        auto action_accessor{actions.accessor<int64_t, 1>()};
        auto step_accessor{steps.accessor<int64_t, 1>()};
        auto terminals_accessor{terminals.accessor<bool, 1>()};

        for (int64_t i = 0; i < options.batchsize; i++) {
            mcts_nodes[i] = mcts_nodes[i]->get_child(action_accessor[i]);

            auto &step = step_accessor[i];
            reward_history.index_put_({i, step}, mcts_nodes[i]->reward());

            step += 1;
            if (step >= options.max_episode_length || mcts_nodes[i]->terminal()) {
                terminals_accessor[i] = true;
            }
            // State and mask are added to the start of next step, if state is not
            // terminal.
            else {
                state_history.index_put_({i, step}, mcts_nodes[i]->state());
                mask_history.index_put_({i, step}, mcts_nodes[i]->mask());
            }
        }

        return terminals;
    }

    void SelfPlayWorker::step()
    {
        mcts(&mcts_nodes, module, simulator, options.mcts_options);
        auto policy = mcts_nodes_to_policy(mcts_nodes, options.temperature_control->get());
        auto actions = policy.sample();
        auto terminals = step_mcts_nodes(actions);

        if (terminals.any().item().toBool()) {
            process_terminals(terminals);
            reset_histories(terminals);
            reset_mcts_nodes(terminals);
        }
    }

    void SelfPlayWorker::worker()
    {
        set_initial_state();

        while (running) {
            step();
        }
    }

    void SelfPlayWorker::process_terminals(const torch::Tensor &terminal_mask)
    {
        auto steps = this->steps.index({terminal_mask});
        auto states = this->state_history.index({terminal_mask});
        auto masks = this->mask_history.index({terminal_mask});
        auto rewards = this->reward_history.index({terminal_mask});

        auto max_length = steps.max().item().toLong();

        auto G = rl::utils::reward::backpropagate(
            rewards.index({Slice(), Slice(None, max_length)}),
            options.discount
        );

        auto batchsize = steps.size(0);

        for (int i = 0; i < batchsize; i++)
        {
            auto episode_length = steps.index({i}).item().toLong();
            SelfPlayEpisode episode{};
            episode.states = states.index({i, Slice(None, episode_length)});
            episode.masks = masks.index({i, Slice(None, episode_length)});
            episode.collected_rewards = G.index({i, Slice(None, episode_length)});

            enqueue_episode(episode);

            if (options.hindsight_callback) {
                SelfPlayEpisode hindsight_episode{};
                hindsight_episode.states = episode.states.clone();
                hindsight_episode.masks = episode.masks.clone();
                hindsight_episode.collected_rewards = episode.collected_rewards.clone();
                auto should_enqueue = options.hindsight_callback(&hindsight_episode);

                if (should_enqueue) {
                    enqueue_episode(hindsight_episode);
                }
            }
        }

        if (options.logger) {
            options.logger->log_scalar("AlphaZero/Reward", G.index({Slice(), 0}).mean().item().toFloat());
            options.logger->log_frequency("AlphaZero/Episode rate", batchsize);
        }
    }

    void SelfPlayWorker::enqueue_episode(const SelfPlayEpisode &episode)
    {
        bool enqueued{false};
        while (!enqueued) {
            enqueued = episode_queue->enqueue(episode, std::chrono::seconds(5));
        }
    }
}
