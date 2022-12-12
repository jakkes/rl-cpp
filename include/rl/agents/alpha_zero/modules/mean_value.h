#ifndef RL_AGENTS_ALPHA_ZERO_MODULES_MEAN_VALUE_H_
#define RL_AGENTS_ALPHA_ZERO_MODULES_MEAN_VALUE_H_


#include <torch/torch.h>

#include "base.h"


namespace rl::agents::alpha_zero::modules
{
    class MeanValueOutput : public BaseOutput
    {
        public:
            MeanValueOutput(
                const torch::Tensor &prior_logits,
                const torch::Tensor &mean_values
            ) : BaseOutput{prior_logits}, values{mean_values}
            {}

            inline
            torch::Tensor value_estimates() const override {
                return values;
            }

            inline
            torch::Tensor value_loss(const torch::Tensor &rewards) const override {
                return (values - rewards).square();
            }
        
        private:
            torch::Tensor values;
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_MODULES_MEAN_VALUE_H_ */
