#ifndef RL_AGENTS_DQN_TRAINERS_BASIC_H_
#define RL_AGENTS_DQN_TRAINERS_BASIC_H_


#include <memory>

#include <torch/torch.h>

#include <rl/option.h>
#include <rl/env/base.h>
#include <rl/logging/client/base.h>
#include <rl/agents/dqn/modules/base.h>
#include <rl/agents/dqn/policies/base.h>

namespace rl::agents::dqn::trainers
{
    struct BasicOptions
    {
        // Number of environment steps to execute per network update step.
        RL_OPTION(int, environment_steps_per_training_step) = 4;
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
        // Number of training steps between each target network update.
        RL_OPTION(int, target_network_update_steps) = 100;
        // Discount factor
        RL_OPTION(float, discount) = 0.99;
        // Logging client
        RL_OPTION(std::shared_ptr<rl::logging::client::Base>, logger) = nullptr;
    };

    /**
     * @brief Basic sequential trainer, executing training steps and environment steps
     * in alternating fashion.
     * 
     */
    class Basic
    {
        public:
            Basic(
                std::shared_ptr<rl::agents::dqn::modules::Base> module,
                std::shared_ptr<rl::agents::dqn::policies::Base> policy,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                std::shared_ptr<rl::env::Factory> env_factory,
                const BasicOptions &options={}
            );

            /**
             * @brief Starts the training process, and blocks until completed.
             * 
             * @param duration Training duration, seconds.
             */
            void run(size_t duration);

        private:
            const BasicOptions options;
            std::shared_ptr<rl::agents::dqn::modules::Base> module;
            std::shared_ptr<rl::agents::dqn::modules::Base> target_module;
            std::shared_ptr<rl::agents::dqn::policies::Base> policy;
            std::shared_ptr<torch::optim::Optimizer> optimizer;
            std::shared_ptr<rl::env::Factory> env_factory;
    };
}

#endif /* RL_AGENTS_DQN_TRAINERS_BASIC_H_ */
