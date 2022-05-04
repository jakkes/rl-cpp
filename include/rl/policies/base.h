#ifndef RL_POLICIES_BASE_H_
#define RL_POLICIES_BASE_H_

#include <memory>
#include <torch/torch.h>

#include "unsupported_constraint_exception.h"
#include "constraints/base.h"
#include "constraints/empty.h"


namespace rl::policies
{
    /**
     * @brief Base policy class.
     * 
     * A policy is nothing other than a probability distribution,
     * 
     */
    class Base {
        public:
            virtual torch::Tensor sample() const = 0;
            virtual torch::Tensor entropy() const = 0;
            virtual torch::Tensor log_prob(const torch::Tensor &value) const = 0;
            virtual torch::Tensor prob(const torch::Tensor &value) const = 0;
            
            virtual void include(std::shared_ptr<constraints::Base> constraint)
            {
                // If no constraint
                if (std::dynamic_pointer_cast<constraints::Empty>(constraint)) {
                    return;
                }

                throw UnsupportedConstraintException{};
            }
            virtual std::unique_ptr<Base> index(const std::vector<torch::indexing::TensorIndex> &indexing) const
            {
                throw std::runtime_error{"Policy index not implemented."};
            }
    };
}

#endif /* RL_POLICIES_BASE_H_ */
