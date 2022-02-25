#ifndef RL_POLICIES_BASE_H_
#define RL_POLICIES_BASE_H_

#include <memory>
#include <torch/torch.h>

#include "unsupported_constraint_exception.h"
#include "constraints/base.h"


namespace rl::policies
{

    class Base{
        public:
            virtual const torch::Tensor sample() const = 0;
            virtual const torch::Tensor log_prob(const torch::Tensor &value) const = 0;
            virtual void include(const constraints::Base &constraint) const = 0;
    };
}

#endif /* RL_POLICIES_BASE_H_ */
