#ifndef RL_AGENTS_PPO_TRAINERS_SEED_IMPL_ACTOR_H_
#define RL_AGENTS_PPO_TRAINERS_SEED_IMPL_ACTOR_H_


#include <vector>
#include <thread>
#include <atomic>
#include <memory>

#include <thread_safe/collections/queue.h>

#include "rl/env/env.h"
#include "rl/option.h"
#include "rl/logging/client/base.h"

#include "inference.h"
#include "sequence.h"

namespace rl::agents::ppo::trainers::seed_impl
{

    struct ActorOptions
    {
        RL_OPTION(int, sequence_length) = 64;
        RL_OPTION(int, environments) = 1;
        RL_OPTION(torch::Device, environment_device) = torch::kCPU;

        RL_OPTION(std::shared_ptr<rl::logging::client::Base>, logger) = nullptr;
    };

    class Actor
    {
        public:
            Actor() = delete;
            Actor(Actor &actor) = delete;
            Actor(
                std::shared_ptr<Inference> inference,
                std::shared_ptr<rl::env::Factory> env_factory,
                std::shared_ptr<thread_safe::Queue<std::shared_ptr<Sequence>>> out_stream,
                const ActorOptions &options={}
            );

            void start();
            void stop();
            void join();
        
        private:
            const ActorOptions options;
            std::shared_ptr<Inference> inference;
            std::shared_ptr<rl::env::Factory> env_factory;
            std::shared_ptr<thread_safe::Queue<std::shared_ptr<Sequence>>> out_stream;

            std::atomic<bool> is_running{false};
            std::thread working_thread;

            void worker();
    };
}

#endif /* RL_AGENTS_PPO_TRAINERS_SEED_IMPL_ACTOR_H_ */
