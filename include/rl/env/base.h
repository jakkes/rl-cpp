#ifndef RL_ENV_BASE_H_
#define RL_ENV_BASE_H_


#include <memory>

#include <torch/torch.h>

#include "observation.h"
#include "state.h"


namespace rl::env
{

    /**
     * @brief Base class for all environments.
     * 
     */
    class Base{
        public:
            virtual std::unique_ptr<const Observation> step(torch::Tensor action);
            virtual std::unique_ptr<const State> reset();

    };
}

#endif /* RL_ENV_BASE_H_ */
