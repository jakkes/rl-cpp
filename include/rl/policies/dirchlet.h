#ifndef RL_POLICIES_DIRCHLET_H_
#define RL_POLICIES_DIRCHLET_H_


#include <memory>

#include <torch/torch.h>

#include "base.h"
#include "gamma.h"

namespace rl::policies
{
    /**
     * @brief Dirchlet policy.
     * 
     * Given {k_1, ..., k_n}, where k_i > 0, the dirchlet policy samples actions
     * {a_1, ..., a_n} such that, sum(a_i) == 1.0 and a_i > 0.
     * 
     */
    class Dirchlet : public Base
    {
        public:
            /**
             * @brief Construct a new Dirchlet policy.
             * 
             * @param coefficients Tensor of shape (**, N), where N denotes the number
             * of output values. All values must be greater than zero.
             */
            Dirchlet(torch::Tensor coefficients);

            torch::Tensor sample() const override;
            torch::Tensor entropy() const override;

            torch::Tensor log_prob(const torch::Tensor &x) const override;
            torch::Tensor prob(const torch::Tensor &value) const override;

            using Base::include;
            std::unique_ptr<Base> index(const std::vector<torch::indexing::TensorIndex> &indexing) const override;

        private:
            torch::Tensor coefficients;
            int64_t dim;
            std::shared_ptr<Gamma> gamma;
    };
}

#endif /* RL_POLICIES_DIRCHLET_H_ */
