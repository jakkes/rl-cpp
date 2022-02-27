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
            virtual torch::Tensor sample() const = 0;
            virtual torch::Tensor log_prob(const torch::Tensor &value) const = 0;
            virtual void include(std::shared_ptr<constraints::Base> constraint)
            {
                throw UnsupportedConstraintException{};
            }
    };
}

#endif /* RL_POLICIES_BASE_H_ */
