#ifndef RL_AGENTS_DQN_TRAINERS_SEED_H_
#define RL_AGENTS_DQN_TRAINERS_SEED_H_


#include <memory>

#include <torch/torch.h>

#include <rl/option.h>
#include <rl/agents/dqn/modules/base.h>
#include <rl/env/base.h>


namespace rl::agents::dqn::trainers
{
    struct SEEDOptions
    {
        // Replay buffer size
        RL_OPTION(int64_t, replay_buffer_size) = 100000;
        // Training is paused until the replay buffer is filled with at least this
        // number of samples.
        RL_OPTION(int64_t, minimum_replay_buffer_size) = 10000;
        // Batch size used in training.
        RL_OPTION(int, batch_size) = 64;
        // Device where replay is located.
        RL_OPTION(torch::Device, replay_device) = torch::kCPU;
        // Device where network is located.
        RL_OPTION(torch::Device, network_device) = torch::kCPU;
        // Device on which environment observations are located.
        RL_OPTION(torch::Device, environment_device) = torch::kCPU;
    };

    class SEED
    {
        public:
            SEED(
                std::shared_ptr<rl::agents::dqn::modules::Base> module,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                std::shared_ptr<rl::env::Factory> env_factory,
                const SEEDOptions &options={}
            );

            void run(int64_t duration_seconds);

        private:
            const SEEDOptions options;
            std::shared_ptr<rl::agents::dqn::modules::Base> module;
            std::shared_ptr<rl::agents::dqn::modules::Base> target_module;
            std::shared_ptr<torch::optim::Optimizer> optimizer;
            std::shared_ptr<rl::env::Factory> env_factory;
    };
}

#endif /* RL_AGENTS_DQN_TRAINERS_SEED_H_ */
