#ifndef RL_ENV_STATE_H_
#define RL_ENV_STATE_H_

#include <torch/torch.h>

#include "constraints/base.h"


namespace rl::env
{
    struct State
    {
        torch::Tensor state;
        std::unique_ptr<constraints::Base> action_constraint;
    };
}

#endif /* RL_ENV_STATE_H_ */
