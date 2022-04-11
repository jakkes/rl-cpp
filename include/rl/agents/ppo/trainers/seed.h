#ifndef RL_AGENTS_PPO_TRAINERS_SEED_H_
#define RL_AGENTS_PPO_TRAINERS_SEED_H_

#include <memory>
#include <chrono>

#include <torch/torch.h>

#include "rl/option.h"
#include "rl/env/env.h"
#include "rl/policies/policies.h"
#include "rl/agents/ppo/module.h"

namespace rl::agents::ppo::trainers
{
    struct SEEDOptions
    {
        RL_OPTION(float, eps) = 0.1;
        RL_OPTION(float, discount) = 0.99;
        RL_OPTION(float, gae_discount) = 0.95;
        
        RL_OPTION(int, sequence_length) = 64;
        RL_OPTION(int, envs_per_worker) = 4;
        RL_OPTION(int, env_workers) = 4;

        RL_OPTION(int, batchsize) = 32;

        RL_OPTION(int, inference_batchsize) = 32;
        RL_OPTION(int, inference_max_delay_ms) = 500;

        RL_OPTION(int64_t, replay_size) = 500;
        RL_OPTION(int64_t, inference_replay_size) = 100;
        RL_OPTION(torch::Device, replay_device) = torch::kCPU;
        RL_OPTION(int64_t, min_replay_size) = 500;
    };

    class SEED
    {
        public:
            SEED(
                std::shared_ptr<rl::agents::ppo::Module> model,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                std::shared_ptr<rl::env::Factory> env_factory,
                const SEEDOptions &options={}
            );

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
