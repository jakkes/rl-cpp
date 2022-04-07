#ifndef RL_ENV_OBSERVATION_H_
#define RL_ENV_OBSERVATION_H_

#include <memory>

#include <torch/torch.h>

#include "state.h"


namespace rl::env
{
    /**
     * @brief Observation of a state transition.
     * 
     */
    struct Observation{
        // Next state of the environment.
        std::unique_ptr<State> state;
        
        // Reward given in the transition.
        float reward;
        
        // Whether or not the environment transitioned into a terminal state.
        bool terminal;
    };
}

#endif /* RL_ENV_OBSERVATION_H_ */
