#include "self_play_worker.h"

#include <c10/cuda/CUDAStream.h>
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

        auto output = inference_fn(states.to(options.module_device));
        auto priors = output.policies.get_probabilities().to(torch::kCPU);
        auto values = output.values.to(torch::kCPU);
        
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
            options.logger->log_scalar("AlphaZero/Start entropy", output.policies.entropy().mean().item().toFloat());
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

        float end_value{0.0f};
        int n_terminals{0};

        for (int64_t i = 0; i < options.batchsize; i++) {
            auto next_node = mcts_nodes[i]->get_child(action_accessor[i]);

            auto &step = step_accessor[i];
            reward_history.index_put_({i, step}, next_node->reward());

            step += 1;
            if (step >= options.max_episode_length || next_node->terminal()) {
                terminals_accessor[i] = true;
                end_value += mcts_nodes[i]->v();
                n_terminals++;
            }
            // State and mask are added to the start of next step, if state is not
            // terminal.
            else {
                state_history.index_put_({i, step}, next_node->state());
                mask_history.index_put_({i, step}, next_node->mask());
            }

            mcts_nodes[i] = next_node;
        }

        if (options.logger && n_terminals > 0) {
            options.logger->log_scalar("AlphaZero/End value", end_value / n_terminals);
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
        torch::StreamGuard stream_guard{c10::cuda::getStreamFromPool()};
        inference_fn_setup();

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

                    if (options.logger) {
                        options.logger->log_scalar(
                            "AlphaZero/Hindsight reward",
                            hindsight_episode.collected_rewards.index({0}).item().toFloat()
                        );
                        options.logger->log_frequency(
                            "AlphaZero/Hindsight episode rate", 1
                        );
                    }
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

    void SelfPlayWorker::inference_fn_setup()
    {
        if (options.module_device.is_cuda()) {
            cuda_graph_inference_setup();
            inference_fn = std::bind(&SelfPlayWorker::cuda_graph_inference_fn, this, std::placeholders::_1);
        }
        else {
            inference_fn = std::bind(&SelfPlayWorker::cpu_inference_fn, this, std::placeholders::_1);
        }
    }

    void SelfPlayWorker::cuda_graph_inference_setup()
    {
        inference_graph = std::make_unique<at::cuda::CUDAGraph>();
        inference_input = simulator->reset(options.batchsize).states.to(options.module_device);

        inference_graph->capture_begin();
        auto module_output = module->forward(inference_input);
        inference_policy_output = module_output->policy().get_probabilities();
        inference_value_output = module_output->value_estimates();
        inference_graph->capture_end();
    }

    MCTSInferenceResult SelfPlayWorker::cuda_graph_inference_fn(const torch::Tensor &states)
    {
        inference_input.fill_(states);
        inference_graph->replay();
        return MCTSInferenceResult{
            inference_policy_output.clone(), inference_value_output.clone()
        };
    }
}
