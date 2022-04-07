#ifndef RL_POLICIES_FACTORIZED_H_
#define RL_POLICIES_FACTORIZED_H_


#include <vector>
#include <memory>

#include <torch/torch.h>

#include "base.h"


namespace rl::policies
{
    class Factorized : public Base
    {
        public:
            Factorized(const std::vector<std::shared_ptr<Base>> &policies);

            torch::Tensor sample() const override;
            torch::Tensor entropy() const override;

            torch::Tensor log_prob(const torch::Tensor &x) const override;
            torch::Tensor prob(const torch::Tensor &value) const override;

        private:
            std::vector<std::shared_ptr<Base>> policies;
    };
}

#endif /* RL_POLICIES_FACTORIZED_H_ */
