#ifndef RL_ENV_STATE_H_
#define RL_ENV_STATE_H_

#include <memory>

#include <torch/torch.h>

#include "rl/policies/constraints/base.h"


namespace rl::env
{
    /**
     * @brief State of an environment.
     * 
     */
    struct State
    {
        // State representation.
        torch::Tensor state;

        // Constraints on the next action.
        std::shared_ptr<rl::policies::constraints::Base> action_constraint;
    };
}

#endif /* RL_ENV_STATE_H_ */
