#ifndef RL_CPP_INCLUDE_RL_POLICIES_CONSTRAINTS_EMPTY_H_
#define RL_CPP_INCLUDE_RL_POLICIES_CONSTRAINTS_EMPTY_H_


#include "base.h"

namespace rl::policies::constraints
{
    /**
     * @brief Constraint class representing an empty set of constraints.
     * 
     */
    class Empty : public Base {
        public:
            Empty(int n_action_dims);

            torch::Tensor contains(const torch::Tensor &x) const;
            std::unique_ptr<Base> index(const std::vector<torch::indexing::TensorIndex> &indexing) const;
            std::unique_ptr<Base> stack(const std::vector<std::shared_ptr<Base>> &constraints) const;

        private:
            const int n_action_dims;
    };
}

#endif /* RL_CPP_INCLUDE_RL_POLICIES_CONSTRAINTS_EMPTY_H_ */
