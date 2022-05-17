#ifndef RL_POLICIES_DIRCHLET_H_
#define RL_POLICIES_DIRCHLET_H_

#include <torch/torch.h>

#include "base.h"
#include "gamma.h"

namespace rl::policies
{
    class Dirchlet : public Base
    {
        public:
            Dirchlet(torch::Tensor coefficients, torch::Tensor a, torch::Tensor b);
            Dirchlet(torch::Tensor coefficients);

            torch::Tensor sample() const override;
            torch::Tensor entropy() const override;

            torch::Tensor log_prob(const torch::Tensor &x) const override;
            torch::Tensor prob(const torch::Tensor &value) const override;

            using Base::include;
            std::unique_ptr<Base> index(const std::vector<torch::indexing::TensorIndex> &indexing) const override;

        private:
            torch::Tensor coefficients, a, b;
            int64_t dim;
            Gamma gamma;
    };
}

#endif /* RL_POLICIES_DIRCHLET_H_ */
