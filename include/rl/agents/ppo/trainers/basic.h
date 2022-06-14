#ifndef INCLUDE_RL_AGENTS_PPO_TRAINERS_BASIC_H_
#define INCLUDE_RL_AGENTS_PPO_TRAINERS_BASIC_H_

#include <memory>
#include <chrono>
#include <functional>
#include <atomic>

#include <torch/torch.h>

#include "rl/env/env.h"
#include "rl/policies/policies.h"
#include "rl/agents/ppo/module.h"
#include "rl/logging/client/base.h"


namespace rl::agents::ppo::trainers
{

    /**
     * @brief Options for the Basic trainer
     */
    struct BasicOptions{
        // Epsilon, controlling policy proximity between training steps.
        RL_OPTION(float, eps) = 0.1;
        // Reward discount factor.
        RL_OPTION(float, discount) = 0.99;
        // Discount factor for generalized advantage estimation.
        RL_OPTION(float, gae_discount) = 0.95;
        // Number of update steps per gathered data batch.
        RL_OPTION(int, update_steps) = 10;
        // Sequence length of gathered data batches
        RL_OPTION(int, sequence_length) = 64;
        // Number of parallel environments from which data is gathered.
        RL_OPTION(int, envs) = 16;
        // Number of threads on which the data collection is distribution on.
        RL_OPTION(int, env_workers) = 4;
        // If true, data is stored on the gpu.
        RL_OPTION(bool, cuda) = false;

        // Logger used by the trainer.
        RL_OPTION(std::shared_ptr<rl::logging::client::Base>, logger){};
        // If true, policy entropy is logged at the initial state.
        RL_OPTION(bool, log_start_entropy) = true;
        // If true, loss function is logged.
        RL_OPTION(bool, log_loss) = true;
        // If true, value estimates of the initial states are logged.
        RL_OPTION(bool, log_start_value) = true;
    };

    /**
     * @brief Basic trainer for PPO models. This trainer is a (close to) replicate of
     * the training algorithm in the original paper.
     */
    class Basic{
        public:
            /**
             * @brief Construct a new Basic object
             * 
             * @param model PPO model
             * @param optimizer Model optimizer
             * @param env_factory Environment factory
             * @param options Trainer options
             */
            Basic(
                std::shared_ptr<rl::agents::ppo::Module> model,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                std::shared_ptr<rl::env::Factory> env_factory,
                const BasicOptions &options={}
            );

            /**
             * @brief Starts the training process, and blocks until completed.
             * 
             * @param duration Training duration.
             */
            template<class Rep, class Period>
            void run(std::chrono::duration<Rep, Period> duration);

        private:
            std::shared_ptr<rl::agents::ppo::Module> model;
            std::shared_ptr<torch::optim::Optimizer> optimizer;
            std::shared_ptr<rl::env::Factory> env_factory;
            BasicOptions options;
    };
}

#endif /* INCLUDE_RL_AGENTS_PPO_TRAINERS_BASIC_H_ */
