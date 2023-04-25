#ifndef RL_AGENTS_DQN_TRAINERS_APEX_H_
#define RL_AGENTS_DQN_TRAINERS_APEX_H_


#include <memory>

#include <torch/torch.h>

#include <rl/option.h>
#include <rl/agents/dqn/modules/base.h>
#include <rl/agents/dqn/policies/base.h>
#include <rl/agents/dqn/utils/hindsight_replay.h>
#include <rl/env/base.h>


namespace rl::agents::dqn::trainers
{
    struct ApexOptions
    {
        // Number of sequences collected by a worker process before added to the replay.
        RL_OPTION(int64_t, inference_replay_size) = 1000;
        // Replay buffer size
        RL_OPTION(int64_t, training_buffer_size) = 100000;
        // Training is paused until the replay buffer is filled with at least this
        // number of samples.
        RL_OPTION(int64_t, minimum_replay_buffer_size) = 10000;
        // Number of env worker threads
        RL_OPTION(int, workers) = 4;
        // Batch size per worker
        RL_OPTION(int, worker_batchsize) = 128;
        // Batch size used in training.
        RL_OPTION(int, batch_size) = 64;
        // Gradients are scaled in case their norm is larger than this value.
        RL_OPTION(float, max_gradient_norm) = 40.0f;
        // Device where replay is located.
        RL_OPTION(torch::Device, replay_device) = torch::kCPU;
        // Device where network is located.
        RL_OPTION(torch::Device, network_device) = torch::kCPU;
        // Device on which environment observations are located.
        RL_OPTION(torch::Device, environment_device) = torch::kCPU;
        // Update rate of target network, value be in (0, 1).
        RL_OPTION(float, target_network_lr) = 1e-3;
        // Discount factor
        RL_OPTION(float, discount) = 0.99;
        // Double DQN
        RL_OPTION(bool, double_dqn) = true;
        // Reward n-step
        RL_OPTION(int, n_step) = 3;
        // Logging client
        RL_OPTION(std::shared_ptr<rl::logging::client::Base>, logger) = nullptr;
        // // Checkpoint callback, called with number of seconds trained so far.
        // RL_OPTION(std::function<void(size_t)>, checkpoint_callback) = nullptr;
        // // Checkpoint callback period, in seconds.
        // RL_OPTION(size_t, checkpoint_callback_period_seconds) = 3600;
        // // If set, this method is called whenever an episode terminates. The argument
        // // is a pointer to (a copy of) the episode. If the callback returns true, then
        // // the (possibly modified sequence) is added to the replay buffer. If false is
        // // returned, then the sequence is not added to the buffer.
        // // 
        // // NOTE: Any tensors modified in place will also modify the actually observed
        // // tensor. Therefore, if states or masks are to be modified, first clone
        // // the tensors.
        // RL_OPTION(rl::agents::dqn::utils::HindsightReplayCallback, hindsight_replay_callback) = nullptr;
    };

    class Apex
    {
        public:
            Apex(
                std::shared_ptr<rl::agents::dqn::modules::Base> module,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                std::shared_ptr<rl::agents::dqn::policies::Base> policy,
                std::shared_ptr<rl::env::Factory> env_factory,
                const ApexOptions &options={}
            );

            void run(int64_t duration_seconds);
        
        private:
            const ApexOptions options;
            std::shared_ptr<rl::agents::dqn::modules::Base> module;
            std::shared_ptr<torch::optim::Optimizer> optimizer;
            std::shared_ptr<rl::agents::dqn::policies::Base> policy;
            std::shared_ptr<rl::env::Factory> env_factory;
    };
}

#endif /* RL_AGENTS_DQN_TRAINERS_APEX_H_ */
