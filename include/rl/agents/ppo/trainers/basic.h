#ifndef INCLUDE_RL_AGENTS_PPO_TRAINERS_BASIC_H_
#define INCLUDE_RL_AGENTS_PPO_TRAINERS_BASIC_H_

#include <memory>
#include <chrono>
#include <functional>
#include <atomic>

#include <torch/torch.h>

#include "rl/cpputils.h"
#include "rl/env/env.h"
#include "rl/policies/policies.h"
#include "rl/agents/ppo/module.h"
#include "rl/logging/client/base.h"


namespace rl::agents::ppo::trainers
{

    struct BasicOptions{
        RL_OPTION(float, eps) = 0.1;
        RL_OPTION(float, discount) = 0.99;
        RL_OPTION(float, gae_discount) = 0.95;
        RL_OPTION(int64_t, update_steps) = 10;
        RL_OPTION(int64_t, sequence_length) = 64;
        RL_OPTION(int, envs) = 16;
        RL_OPTION(int, env_workers) = 4;
        RL_OPTION(bool, cuda) = false;

        RL_OPTION(std::shared_ptr<rl::logging::client::Base>, logger){};
    };

    class Basic{
        public:
            Basic(
                std::shared_ptr<rl::agents::ppo::Module> model,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                std::shared_ptr<rl::env::Factory> env_factory,
                const BasicOptions &options={}
            );

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
