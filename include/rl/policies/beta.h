#ifndef RL_POLICIES_BETA_H_
#define RL_POLICIES_BETA_H_


#include <torch/torch.h>

#include "base.h"


namespace rl::policies
{
    class Beta : public Base
    {
        public:
            Beta(torch::Tensor alpha, torch::Tensor beta, torch::Tensor a, torch::Tensor b);
            Beta(torch::Tensor alpha, torch::Tensor beta);

            torch::Tensor sample() const override;
            torch::Tensor entropy() const override;

            torch::Tensor log_prob(const torch::Tensor &x) const override;
            torch::Tensor prob(const torch::Tensor &value) const override;

            void include(std::shared_ptr<constraints::Base> constraint);

        private:
            torch::Tensor alpha, beta, a, b, c, cdf, pdf, x;
            std::vector<int64_t> sample_shape;
    };
}

#endif /* RL_POLICIES_BETA_H_ */
