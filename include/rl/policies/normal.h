#ifndef RL_POLICIES_NORMAL_H_
#define RL_POLICIES_NORMAL_H_


#include <torch/torch.h>

#include "base.h"



namespace rl::policies
{
    class Normal : public Base
    {
        public:
            Normal(const torch::Tensor &mean, const torch::Tensor &std);
            ~Normal() = default;

            torch::Tensor sample() const override;
            torch::Tensor entropy() const override;

            torch::Tensor log_prob(const torch::Tensor &x) const override;
            torch::Tensor prob(const torch::Tensor &value) const override;

            void include(std::shared_ptr<constraints::Base> constraint);

        private:
            torch::Tensor mean, std;
            torch::Tensor lower_bound, upper_bound;
            torch::Tensor cdf_lower_bound, cdf_upper_bound;

            void compute_cdf_at_bounds();
            torch::Tensor standardize(const torch::Tensor &x) const;
    };
}

#endif /* RL_POLICIES_NORMAL_H_ */
