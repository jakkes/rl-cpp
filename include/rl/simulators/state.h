#ifndef RL_SIMULATORS_STATE_H_
#define RL_SIMULATORS_STATE_H_


#include <torch/torch.h>

namespace rl::simulators
{
    struct States
    {
        torch::Tensor states;
        std::shared_ptr<rl::policies::constraints::Base> action_constraints;
    };
}

#endif /* RL_SIMULATORS_STATE_H_ */
