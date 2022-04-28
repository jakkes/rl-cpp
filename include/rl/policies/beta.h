#ifndef RL_POLICIES_BETA_H_
#define RL_POLICIES_BETA_H_


#include <torch/torch.h>

#include "base.h"
#include "gamma.h"


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

            void include(std::shared_ptr<constraints::Base> constraint) override;
            std::unique_ptr<Base> index(const std::vector<torch::indexing::TensorIndex> &indexing) const override;

        private:
            const Gamma x, y;
            const torch::Tensor a, b, alpha, beta;
    };
}

#endif /* RL_POLICIES_BETA_H_ */
