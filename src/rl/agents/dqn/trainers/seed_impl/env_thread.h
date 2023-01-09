#ifndef RL_AGENTS_DQN_TRAINERS_SEED_IMPL_ACTOR_H_
#define RL_AGENTS_DQN_TRAINERS_SEED_IMPL_ACTOR_H_


#include <vector>
#include <memory>
#include <thread>
#include <atomic>

#include <torch/torch.h>
#include <thread_safe/collections/queue.h>

#include <rl/env/base.h>
#include <rl/agents/dqn/trainers/seed.h>
#include <rl/utils/reward/n_step_collector.h>
#include <rl/agents/dqn/utils/hindsight_replay.h>

#include "inferer.h"

namespace seed_impl
{
    class EnvWorker
    {
        public:
            EnvWorker(
                std::shared_ptr<rl::env::Factory> env_factory,
                std::shared_ptr<Inferer> inferer,
                std::shared_ptr<thread_safe::Queue<rl::utils::reward::NStepCollectorTransition>> transition_queue,
                const rl::agents::dqn::trainers::SEEDOptions &options
            );

            inline
            bool ready() const { return result_future->ready(); }

            void step();

        private:
            const rl::agents::dqn::trainers::SEEDOptions options;
            std::shared_ptr<Inferer> inferer;
            std::shared_ptr<thread_safe::Queue<rl::utils::reward::NStepCollectorTransition>> transition_queue;
            std::unique_ptr<rl::env::Base> env;

            rl::utils::reward::NStepCollector n_step_collector;

            std::shared_ptr<rl::env::State> state;
            std::unique_ptr<InferenceResultFuture> result_future;

            bool start_state{true};
            std::unique_ptr<rl::agents::dqn::utils::HindsightReplayEpisode> episode;
        
        private:
            void process_episode();
    };

    class EnvThread
    {
        public:
            EnvThread(
                std::shared_ptr<rl::env::Factory> env_factory,
                std::shared_ptr<Inferer> inferer,
                std::shared_ptr<thread_safe::Queue<rl::utils::reward::NStepCollectorTransition>> transition_queue,
                const rl::agents::dqn::trainers::SEEDOptions &options
            );

            void start();
            void stop();

        private:
            rl::agents::dqn::trainers::SEEDOptions options;
            std::shared_ptr<Inferer> inferer;
            std::shared_ptr<rl::env::Factory> env_factory;
            std::shared_ptr<thread_safe::Queue<rl::utils::reward::NStepCollectorTransition>> transition_queue;

            std::atomic<bool> running{false};
            std::thread worker_thread;

            std::vector<EnvWorker> workers{};

        private:
            void worker();
    };
}

#endif /* RL_AGENTS_DQN_TRAINERS_SEED_IMPL_ACTOR_H_ */
