#ifndef RL_AGENTS_PPO_TRAINERS_SEED_H_
#define RL_AGENTS_PPO_TRAINERS_SEED_H_

#include <memory>
#include <chrono>

#include <torch/torch.h>

#include "rl/option.h"
#include "rl/env/env.h"
#include "rl/policies/policies.h"
#include "rl/agents/ppo/module.h"
#include "rl/logging/logging.h"

namespace rl::agents::ppo::trainers
{
    /**
     * @brief Options for the SEED trainer.
     */
    struct SEEDOptions
    {
        // Epsilon, controlling policy proximity between training steps.
        RL_OPTION(float, eps) = 0.1;
        // Reward discount factor.
        RL_OPTION(float, discount) = 0.99;
        // Discount factor for generalized advantage estimation.
        RL_OPTION(float, gae_discount) = 0.95;
        
        // Sequence length of each data batch gathered from the environment.
        RL_OPTION(int, sequence_length) = 64;
        // Number of parallel environments run per worker thread.
        RL_OPTION(int, envs_per_worker) = 4;
        // Number of environment worker threads.
        RL_OPTION(int, env_workers) = 4;
        // Device on which environments expect action tensors.
        RL_OPTION(torch::Device, environment_device) = torch::kCPU;

        // Maximum batch size allowed by the inference worker.
        RL_OPTION(int, inference_batchsize) = 32;
        // Maximum delay allowed by the inference worker.
        RL_OPTION(int, inference_max_delay_ms) = 500;
        // Device on which the network resides.
        RL_OPTION(torch::Device, network_device) = torch::kCPU;

        // Batch size used in training.
        RL_OPTION(int, batchsize) = 32;
        // Size of replay, in number of sequences stored.
        RL_OPTION(int64_t, replay_size) = 2000;
        // Number of sequences collected by the inference process before added to the replay.
        RL_OPTION(int64_t, inference_replay_size) = 500;
        // Device on which data is stored.
        RL_OPTION(torch::Device, replay_device) = torch::kCPU;
        // Minimum replay size before training starts.
        RL_OPTION(int64_t, min_replay_size) = 1000;
        // Upper bound on number of training steps executed per second.
        RL_OPTION(float, max_update_frequency) = 10;
        // Value loss is multiplied by this coefficient.
        RL_OPTION(float, value_loss_coefficient) = 1.0;
        // Policy loss is multiplied by this coefficient.
        RL_OPTION(float, policy_loss_coefficient) = 1.0;
        // Entropy loss is multipled by this coefficient. Note, that positive values
        // imply that entropy is to be maximized.
        RL_OPTION(float, entropy_loss_coefficient) = 0.0;

        // Logger used by the trainer.
        RL_OPTION(std::shared_ptr<rl::logging::client::Base>, logger) = nullptr;

        // Checkpoint callback. This function is called periodically. During its
        // execution, the training is paused, allowing the user to save the current
        // network state.
        RL_OPTION(std::function<void()>, callback) = nullptr;

        // Checkpoint callback period, seconds.
        RL_OPTION(size_t, callback_period) = 1800;
    };

    /**
     * @brief Distributed and scalable trainer based on https://arxiv.org/abs/1910.06591.
     * 
     * The SEED trainer consists of three components, a training thread, an inference
     * thread, and multiple actor threads.
     * 
     * The training thread operates on a small replay buffer. The replay buffer is small
     * to make sure that only recent samples are used in the training. Data is fed into
     * this buffer by the inference thread.
     * 
     * Actor threads hold (possibly) multiple environment instances, and send inference
     * requests to the inference thread. Inference responses, i.e. actions, are then
     * executed in the environment.
     * 
     * The inference thread responds to inference requests and stores data into an
     * inference replay buffer. Whenever this buffer becomes full, its data is moved
     * into the replay buffer used by the training process.
     */
    class SEED
    {
        public:
            /**
             * @brief Construct a new SEED object
             * 
             * @param model PPO Model
             * @param optimizer Model optimizer
             * @param env_factory Environment factory
             * @param options SEED options
             */
            SEED(
                std::shared_ptr<rl::agents::ppo::Module> model,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                std::shared_ptr<rl::env::Factory> env_factory,
                const SEEDOptions &options={}
            );

            /**
             * @brief Starts the training and blocks until completed.
             * 
             * @param duration training duration.
             */
            template<class Rep, class Period>
            void run(std::chrono::duration<Rep, Period> duration);

        private:
            std::shared_ptr<rl::agents::ppo::Module> model;
            std::shared_ptr<torch::optim::Optimizer> optimizer;
            std::shared_ptr<rl::env::Factory> env_factory;
            SEEDOptions options;
    };
}

#endif /* RL_AGENTS_PPO_TRAINERS_SEED_H_ */
