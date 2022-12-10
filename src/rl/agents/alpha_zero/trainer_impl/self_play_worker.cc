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

    void SelfPlayWorker::step_mcts_nodes(const torch::Tensor &actions)
    {
        auto action_accessor{actions.accessor<int64_t, 1>()};

        std::vector<int64_t> null_node_indices{};
        null_node_indices.reserve(options.batchsize);

        std::vector<torch::Tensor> null_node_states{};
        null_node_states.reserve(options.batchsize);

        for (int i = 0; i < options.batchsize; i++) {
            auto next_node = mcts_nodes[i]->get_child(action_accessor[i]);
            
            if (next_node) {
                mcts_nodes[i] = mcts_nodes[i]->get_child(action_accessor[i]);
            }
            else {
                null_node_indices.push_back(i);
                null_node_states.push_back(mcts_nodes[i]->state());
            }
        }

        if (null_node_states.empty()) {
            return;
        }

        auto observation = simulator->step(
            torch::stack(null_node_states, 0),
            actions.index({
                torch::from_blob(
                    null_node_indices.data(),
                    {static_cast<int64_t>(null_node_indices.size())},
                    torch::TensorOptions{}.dtype(torch::kLong)
                )
            })
        );

        auto module_output = module->forward(observation.next_states.states);
        auto next_mask = get_mask(*observation.next_states.action_constraints);
        auto next_priors = module_output->policy().get_probabilities();
        auto next_values = module_output->value_estimates();

        for (int i = 0; i < null_node_indices.size(); i++) {
            mcts_nodes[null_node_indices[i]]->expand(
                action_accessor[null_node_indices[i]],
                observation.rewards.index({i}).item().toFloat(),
                observation.terminals.index({i}).item().toBool(),
                observation.next_states.states.index({i}),
                next_mask.index({i}),
                next_priors.index({i}),
                next_values.index({i}),
                options.mcts_options
            );

            mcts_nodes[null_node_indices[i]] = mcts_nodes[null_node_indices[i]]->get_child(action_accessor[null_node_indices[i]]);
        }
    }

    void SelfPlayWorker::step()
    {
        mcts(&mcts_nodes, module, simulator, options.mcts_options);

        auto masks = mask_history.index({batchvec, steps});
        auto policy = mcts_nodes_to_policy(mcts_nodes, masks, options.temperature);

        auto actions = policy.sample();
        step_mcts_nodes(actions);

        auto states = state_history.index({batchvec, steps});
        auto observations = simulator->step(states, actions);

        reward_history = reward_history.index_put({batchvec, steps}, observations.rewards);
        
        steps += 1;
        observations.terminals = observations.terminals.logical_or(steps >= options.max_episode_length);

        bool any_terminals = observations.terminals.any().item().toBool();
        if (any_terminals) {
            process_terminals(observations.terminals);
        }

        state_history.index_put_({batchvec, steps}, observations.next_states.states);
        mask_history.index_put_(
            {batchvec, steps},
            get_mask(*observations.next_states.action_constraints)
        );

        if (any_terminals) {
            reset_histories(observations.terminals);
            reset_mcts_nodes(observations.terminals);
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

            bool enqueued{false};
            while (!enqueued) {
                enqueued = episode_queue->enqueue(episode, std::chrono::seconds(5));
            }
        }

        this->steps.index_put_({terminal_mask}, 0);

        if (options.logger) {
            options.logger->log_scalar("AlphaZero/Reward", G.index({Slice(), 0}).mean().item().toFloat());
        }
    }
}
