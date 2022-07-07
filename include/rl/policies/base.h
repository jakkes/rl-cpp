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
     * A policy is nothing other than a probability distribution over an action space.
     * 
     */
    class Base : public torch::nn::Module {

        public:
            /**
             * @brief Sample an action from the policy.
             * 
             * @return torch::Tensor Action
             */
            virtual torch::Tensor sample() const = 0;

            /**
             * @brief Entropy of the policy.
             * 
             * @return torch::Tensor Entropy.
             */
            virtual torch::Tensor entropy() const = 0;

            /**
             * @brief Logarithmic likelihood of observing the action under the given
             * policy.
             * 
             * @param action Action.
             * @return torch::Tensor Log likelihood.
             */
            virtual torch::Tensor log_prob(const torch::Tensor &action) const = 0;

            /**
             * @brief Likelihood of observing the action under the given policy.
             * 
             * @param action Action.
             * @return torch::Tensor Likelihood.
             */
            virtual torch::Tensor prob(const torch::Tensor &action) const = 0;
            
            /**
             * @brief Include a policy constraint into the policy, possibly modifying
             * its probability distribution.
             * 
             * @param constraint 
             */
            virtual void include(std::shared_ptr<constraints::Base> constraint)
            {
                // If no constraint
                if (std::dynamic_pointer_cast<constraints::Empty>(constraint)) {
                    return;
                }

                throw UnsupportedConstraintException{};
            }

            /**
             * @brief Index the policy, in case the policy holds batches of policies.
             * 
             * @param indexing Indexing, equivalent to indexing a tensor.
             * @return std::unique_ptr<Base> Indexed policy.
             */
            virtual std::unique_ptr<Base> index(const std::vector<torch::indexing::TensorIndex> &indexing) const
            {
                throw std::runtime_error{"Policy index not implemented."};
            }
    };
}

#endif /* RL_POLICIES_BASE_H_ */
