#ifndef RL_SIMULATORS_BASE_H_
#define RL_SIMULATORS_BASE_H_


#include <memory>

#include <torch/torch.h>

#include <rl/policies/base.h>
#include <rl/logging/client/base.h>

#include "state.h"
#include "observation.h"

namespace rl::simulators
{
    class Base
    {
        public:
            virtual ~Base() = default;

            virtual
            States reset(int n) const = 0;

            virtual
            Observations step(const torch::Tensor &states, const torch::Tensor &actions) const = 0;
    };

    class Factory
    {
        public:
            virtual std::unique_ptr<Base> get() const = 0;
    };
}

#endif /* RL_SIMULATORS_BASE_H_ */
