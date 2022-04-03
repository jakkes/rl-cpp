#ifndef RL_POLICIES_GAMMA_H_
#define RL_POLICIES_GAMMA_H_


#include <torch/torch.h>

#include "base.h"


namespace rl::policies
{



    class Gamma : public Base
    {
        public:
            Gamma(const torch::Tensor &alpha, const torch::Tensor &scale);

            torch::Tensor sample() const override;
            torch::Tensor entropy() const override;

            torch::Tensor log_prob(const torch::Tensor &x) const override;
            torch::Tensor prob(const torch::Tensor &value) const override;

            void include(std::shared_ptr<constraints::Base> constraint);

        private:
            const torch::Tensor _alpha, _is_alpha_boosted, _scale, _d, _c;
            const std::vector<int64_t> shape;
            
    };
}

#endif /* RL_POLICIES_GAMMA_H_ */
