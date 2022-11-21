#ifndef RL_SIMULATORS_OBSERVATION_H_
#define RL_SIMULATORS_OBSERVATION_H_


#include <torch/torch.h>

#include "state.h"

namespace rl::simulators
{
    struct Observations
    {
        States next_states;
        torch::Tensor rewards;
        torch::Tensor terminals;
    };
}

#endif /* RL_SIMULATORS_OBSERVATION_H_ */
