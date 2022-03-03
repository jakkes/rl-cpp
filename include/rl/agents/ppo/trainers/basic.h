#ifndef INCLUDE_RL_AGENTS_PPO_TRAINERS_BASIC_H_
#define INCLUDE_RL_AGENTS_PPO_TRAINERS_BASIC_H_

#include <memory>
#include <chrono>
#include <functional>

#include <torch/torch.h>

#include "rl/cpputils.h"
#include "rl/env/env.h"
#include "rl/policies/policies.h"


namespace rl::agents::ppo::trainers
{

    struct BasicOptions{
        RL_OPTION(int64_t, buffer_size) = 1000;
        RL_OPTION(float, eps) = 0.1;
        RL_OPTION(float, discount) = 0.99;
        RL_OPTION(float, gae_discount) = 0.95;
        RL_OPTION(int64_t, batchsize) = 32;
        RL_OPTION(int64_t, epochs) = 10;
        RL_OPTION(int64_t, sequence_length) = 64;
    };

    class Basic{
        public:
            Basic(
                torch::nn::Module model,
                std::function<std::unique_ptr<rl::policies::Base>(torch::Tensor)> policy_fn,
                std::unique_ptr<torch::optim::Optimizer> optimizer,
                std::unique_ptr<rl::env::Base> env,
                const BasicOptions &options={}
            );

            template<class Rep, class Period>
            void run(std::chrono::duration<Rep, Period> duration);

        private:
            torch::nn::Module model;
            std::function<std::unique_ptr<rl::policies::Base>(torch::Tensor)> policy_fn;
            std::unique_ptr<torch::optim::Optimizer> optimizer;
            std::unique_ptr<rl::env::Base> env;
            BasicOptions options;
    };
}

#endif /* INCLUDE_RL_AGENTS_PPO_TRAINERS_BASIC_H_ */
