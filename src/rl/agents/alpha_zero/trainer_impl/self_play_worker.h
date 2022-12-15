#ifndef RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_SELF_PLAY_WORKER_H_
#define RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_SELF_PLAY_WORKER_H_


#include <atomic>
#include <thread>
#include <vector>
#include <thread_safe/collections/queue.h>

#include <torch/torch.h>

#include <rl/option.h>
#include <rl/simulators/base.h>
#include <rl/agents/alpha_zero/alpha_zero.h>
#include <rl/logging/client/base.h>

#include "self_play_episode.h"

using namespace rl::agents::alpha_zero;

namespace trainer_impl
{
    struct SelfPlayWorkerOptions
    {
        RL_OPTION(int, batchsize) = 32;
        RL_OPTION(int, max_episode_length) = 100;
        RL_OPTION(std::shared_ptr<rl::utils::float_control::Base>, temperature_control) = std::make_shared<rl::utils::float_control::Fixed>(1.0f);
        RL_OPTION(float, discount) = 1.0f;
        RL_OPTION(MCTSOptions, mcts_options) = MCTSOptions{};

        RL_OPTION(std::shared_ptr<rl::logging::client::Base>, logger) = nullptr;
    };

    class SelfPlayWorker
    {
        public:
            SelfPlayWorker() = default;
            SelfPlayWorker(
                std::shared_ptr<rl::simulators::Base> simulator,
                std::shared_ptr<modules::Base> module,
                std::shared_ptr<thread_safe::Queue<SelfPlayEpisode>> episode_queue,
                const SelfPlayWorkerOptions &options={}
            );

            void start();
            void stop();

        private:
            std::shared_ptr<rl::simulators::Base> simulator;
            std::shared_ptr<modules::Base> module;
            std::shared_ptr<thread_safe::Queue<SelfPlayEpisode>> episode_queue;
            const SelfPlayWorkerOptions options;

            std::atomic<bool> running{false};
            std::thread working_thread;

            std::vector<std::shared_ptr<MCTSNode>> mcts_nodes;
            torch::Tensor batchvec;
            torch::Tensor state_history;
            torch::Tensor mask_history;
            torch::Tensor reward_history;
            torch::Tensor steps;

        private:
            void worker();
            void step();
            void step_mcts_nodes(const torch::Tensor &actions);
            void set_initial_state();
            void reset_mcts_nodes(const torch::Tensor &terminal_mask);
            void reset_histories(const torch::Tensor &terminal_mask);
            void process_terminals(const torch::Tensor &terminal_mask);
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_SELF_PLAY_WORKER_H_ */
