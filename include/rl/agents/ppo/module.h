#ifndef INCLUDE_RL_AGENTS_PPO_MODULE_H_
#define INCLUDE_RL_AGENTS_PPO_MODULE_H_

#include <memory>

#include <torch/torch.h>

#include "rl/policies/base.h"


namespace rl::agents::ppo
{

    struct ModuleOutput
    {
        std::unique_ptr<rl::policies::Base> policy;
        torch::Tensor value;
    };

    class Module : public torch::nn::Module
    {
        public:
            virtual std::unique_ptr<ModuleOutput> forward(const torch::Tensor &input) const = 0;
    };
}

#endif /* INCLUDE_RL_AGENTS_PPO_MODULE_H_ */
