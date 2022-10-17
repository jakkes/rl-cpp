#ifndef RL_AGENTS_DQN_TRAINERS_SEED_IMPL_ACTOR_H_
#define RL_AGENTS_DQN_TRAINERS_SEED_IMPL_ACTOR_H_


#include <vector>
#include <memory>
#include <thread>
#include <atomic>

#include <torch/torch.h>

#include <rl/env/base.h>
#include <rl/agents/dqn/trainers/seed.h>

#include "inferer.h"

namespace seed_impl
{
    class EnvWorker
    {
        public:
            EnvWorker(
                std::shared_ptr<rl::env::Factory> env_factory,
                std::shared_ptr<Inferer> inferer,
                const rl::agents::dqn::trainers::SEEDOptions &options
            );

        private:
            const rl::agents::dqn::trainers::SEEDOptions options;
            std::shared_ptr<Inferer> inferer;
            std::unique_ptr<rl::env::Base> env;

            std::unique_ptr<rl::env::State> state;
            InferenceResult result;
    };

    class EnvThread
    {
        public:
            EnvThread(
                std::shared_ptr<rl::env::Factory> env_factory,
                std::shared_ptr<Inferer> inferer,
                const rl::agents::dqn::trainers::SEEDOptions &options
            );

            void start();
            void stop();

        private:
            const rl::agents::dqn::trainers::SEEDOptions options;
            std::shared_ptr<Inferer> inferer;
            std::shared_ptr<rl::env::Factory> env_factory;

            std::atomic<bool> running{false};
            std::thread worker_thread;

            std::vector<EnvWorker> workers{};

        private:
            void worker();
    };
}

#endif /* RL_AGENTS_DQN_TRAINERS_SEED_IMPL_ACTOR_H_ */
