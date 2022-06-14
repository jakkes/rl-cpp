#ifndef INCLUDE_RL_AGENTS_PPO_MODULE_H_
#define INCLUDE_RL_AGENTS_PPO_MODULE_H_

#include <memory>

#include <torch/torch.h>

#include "rl/policies/base.h"


namespace rl::agents::ppo
{

    /**
     * @brief Output of a PPO model
     * 
     * A PPO model outputs an action policy and a value estimate. The policy may not
     * adhere to the action constraints imposed by the environment in its current state.
     * It is up to the user of the policy to impose these.
     */
    struct ModuleOutput
    {
        // Policy
        std::unique_ptr<rl::policies::Base> policy;

        // Value estimate
        torch::Tensor value;
    };

    /**
     * @brief Base class for PPO models.
     */
    class Module : public torch::nn::Module
    {
        public:
            /**
             * @brief Executes the model.
             * 
             * @param input environment state
             * @return std::unique_ptr<ModuleOutput>  model output.
             */
            virtual std::unique_ptr<ModuleOutput> forward(const torch::Tensor &input) = 0;
    };
}

#endif /* INCLUDE_RL_AGENTS_PPO_MODULE_H_ */
