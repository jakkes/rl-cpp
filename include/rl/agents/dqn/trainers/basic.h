#ifndef RL_AGENTS_DQN_TRAINERS_BASIC_H_
#define RL_AGENTS_DQN_TRAINERS_BASIC_H_


#include <memory>
#include <functional>

#include <torch/torch.h>

#include <rl/option.h>
#include <rl/env/base.h>
#include <rl/logging/client/base.h>
#include <rl/agents/dqn/module.h>
#include <rl/agents/dqn/value_parsers/base.h>
#include <rl/agents/dqn/policies/base.h>
#include <rl/agents/dqn/utils/hindsight_replay.h>


namespace rl::agents::dqn::trainers
{

    struct BasicOptions
    {
        // Number of environment steps to execute per network update step.
        RL_OPTION(float, environment_steps_per_training_step) = 1.0f;
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
        RL_OPTION(float, target_network_lr) = 1e-3;
        // Discount factor
        RL_OPTION(float, discount) = 0.99;
        // Double DQN
        RL_OPTION(bool, double_dqn) = true;
        // Reward n-step
        RL_OPTION(int, n_step) = 3;
        // If gradient (L2) norm is larger, then normalize to this value.
        RL_OPTION(float, max_gradient_norm) = 40.0f;
        // Logging client
        RL_OPTION(std::shared_ptr<rl::logging::client::Base>, logger) = nullptr;
        // Checkpoint callback, called with number of training steps executed.
        RL_OPTION(std::function<void(size_t)>, checkpoint_callback) = nullptr;
        // Checkpoint callback period, in number of training steps.
        RL_OPTION(size_t, checkpoint_callback_period) = 100000ul;
        // If set, this method is called whenever an episode terminates. The argument
        // is a pointer to (a copy of) the episode. If the callback returns true, then
        // the (possibly modified sequence) is added to the replay buffer. If false is
        // returned, then the sequence is not added to the buffer.
        RL_OPTION(rl::agents::dqn::utils::HindsightReplayCallback, hindsight_replay_callback) = nullptr;
        // If true, and cuda is used, run in cuda graph mode.
        RL_OPTION(bool, enable_cuda_graph) = true;
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
                std::shared_ptr<rl::agents::dqn::Module> module,
                std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser,
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
            std::shared_ptr<rl::agents::dqn::Module> module;
            std::shared_ptr<rl::agents::dqn::Module> target_module;
            std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser;
            std::shared_ptr<rl::agents::dqn::policies::Base> policy;
            std::shared_ptr<torch::optim::Optimizer> optimizer;
            std::shared_ptr<rl::env::Factory> env_factory;
    };
}

#endif /* RL_AGENTS_DQN_TRAINERS_BASIC_H_ */
