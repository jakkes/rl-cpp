#ifndef RL_ENV_OBSERVATION_H_
#define RL_ENV_OBSERVATION_H_

#include <memory>

#include <torch/torch.h>

#include "state.h"


namespace rl::env
{
    struct Observation{
        std::unique_ptr<State> state;
        float reward;
        bool terminal;
    };
}

#endif /* RL_ENV_OBSERVATION_H_ */
