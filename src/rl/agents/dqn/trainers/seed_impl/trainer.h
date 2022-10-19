#ifndef RL_AGENTS_DQN_TRAINERS_SEED_IMPL_TRAINER_H_
#define RL_AGENTS_DQN_TRAINERS_SEED_IMPL_TRAINER_H_


#include <atomic>
#include <thread>

#include <rl/agents/dqn/trainers/seed.h>
#include <rl/buffers/tensor.h>
#include <rl/buffers/samplers/uniform.h>

using namespace rl::agents::dqn::trainers;

namespace seed_impl
{
    class Trainer
    {
        public:
            Trainer(
                std::shared_ptr<rl::agents::dqn::modules::Base> module,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                std::shared_ptr<rl::env::Factory> env_factory,
                std::shared_ptr<rl::buffers::samplers::Uniform<rl::buffers::Tensor>> sampler,
                const SEEDOptions &options
            );

            void start();
            void stop();

        private:
            const SEEDOptions options;
            std::shared_ptr<rl::agents::dqn::modules::Base> module;
            std::shared_ptr<rl::agents::dqn::modules::Base> target_module;
            std::shared_ptr<torch::optim::Optimizer> optimizer;
            std::shared_ptr<rl::env::Factory> env_factory;
            std::shared_ptr<rl::buffers::samplers::Uniform<rl::buffers::Tensor>> sampler;

            std::atomic<bool> running{false};
            std::thread training_thread;

        private:
            void worker();
            void step();
            void target_network_update();
    };
}

#endif /* RL_AGENTS_DQN_TRAINERS_SEED_IMPL_TRAINER_H_ */
