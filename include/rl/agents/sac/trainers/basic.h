#ifndef RL_AGENTS_SAC_TRAINERS_BASIC_H_
#define RL_AGENTS_SAC_TRAINERS_BASIC_H_


#include <memory>

#include <torch/torch.h>

#include <rl/option.h>
#include <rl/agents/sac/actor.h>
#include <rl/agents/sac/critic.h>
#include <rl/env/base.h>
#include <rl/buffers/tensor.h>
#include <rl/buffers/samplers/uniform.h>
#include <rl/policies/constraints/box.h>

namespace rl::agents::sac::trainers
{
    struct BasicOptions
    {
        // Exploration parameter, higher temperature implies more exploration.
        RL_OPTION(float, temperature) = 0.01f;

        // Actions are scaled to a bounded range using tanh.
        RL_OPTION(float, action_range_min) = -1.0f;
        // Actions are scaled to a bounded range using tanh.
        RL_OPTION(float, action_range_max) = 1.0f;
        
        // Delta used in the Huber loss function for value functions.
        RL_OPTION(float, huber_loss_delta) = 2.0f;
        // If the gradient norm is larger than this value, then it is normed to this 
        // value. Note, the loggers log the unnormed value.
        RL_OPTION(float, max_gradient_norm) = 40.0f;
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
        // Target network learning rate, in (0, 1].
        RL_OPTION(float, target_network_lr) = 1e-3;
        // Discount factor
        RL_OPTION(float, discount) = 0.99;
        // Logging client
        RL_OPTION(std::shared_ptr<rl::logging::client::Base>, logger) = nullptr;
        // Checkpoint callback, called with number of training steps executed.
        RL_OPTION(std::function<void(size_t)>, checkpoint_callback) = nullptr;
        // Checkpoint callback period, in number of training steps.
        RL_OPTION(size_t, checkpoint_callback_period) = 100000ul;
    };

    class Basic
    {
        public:
            Basic(
                std::shared_ptr<rl::agents::sac::Actor> actor,
                std::vector<std::shared_ptr<rl::agents::sac::Critic>> critics,
                std::shared_ptr<torch::optim::Optimizer> actor_optimizer,
                std::vector<std::shared_ptr<torch::optim::Optimizer>> critic_optimizers,
                std::shared_ptr<rl::env::Factory> env_factory,
                const BasicOptions &options={}
            );

            void run(size_t duration);
        
        private:
            const BasicOptions options;
            const std::shared_ptr<rl::agents::sac::Actor> actor;
            const std::shared_ptr<rl::agents::sac::Actor> actor_target;
            const std::vector<std::shared_ptr<rl::agents::sac::Critic>> critics;
            const std::shared_ptr<torch::optim::Optimizer> actor_optimizer;
            const std::vector<std::shared_ptr<torch::optim::Optimizer>> critic_optimizers;
            const std::shared_ptr<rl::env::Factory> env_factory;

            std::shared_ptr<rl::buffers::Tensor> buffer;
            std::shared_ptr<rl::buffers::samplers::Uniform<rl::buffers::Tensor>> sampler;
            std::shared_ptr<rl::env::Base> env;
            size_t env_steps{0};
            size_t train_steps{0};

        private:
            void init_env();
            void initialize_buffer();
            torch::Tensor u_to_a(const torch::Tensor &u);
            torch::Tensor a_to_u(const torch::Tensor &a);
            torch::Tensor log_pi_a(const torch::Tensor &u, const ActorOutput &actor_output);
            void execute_env_step();
            void execute_train_step();
    };
}

#endif /* RL_AGENTS_SAC_TRAINERS_BASIC_H_ */
