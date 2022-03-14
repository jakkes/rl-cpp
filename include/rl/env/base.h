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
            virtual std::unique_ptr<const Observation> step(torch::Tensor action) = 0;
            virtual std::unique_ptr<const State> reset() = 0;
            virtual std::unique_ptr<const State> state() = 0;
            virtual bool is_terminal() = 0;
    };

    class Factory {
        public:
            virtual std::unique_ptr<Base> get() const = 0;
    };
}

#endif /* RL_ENV_BASE_H_ */
