#ifndef RL_SIMULATORS_TRANSITION_H_
#define RL_SIMULATORS_TRANSITION_H_


#include <torch/torch.h>

#include <rl/policies/constraints/base.h>


namespace rl::simulators
{
    struct Transition
    {
        torch::Tensor state;
        std::shared_ptr<rl::policies::constraints::Base> action_constraint;
        float reward;
        bool terminal; 
    };
}

#endif /* RL_SIMULATORS_TRANSITION_H_ */
