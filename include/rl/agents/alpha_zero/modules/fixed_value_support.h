#ifndef RL_AGENTS_ALPHA_ZERO_MODULES_FIXED_SUPPORT_H_
#define RL_AGENTS_ALPHA_ZERO_MODULES_FIXED_SUPPORT_H_


#include <torch/torch.h>

#include "base.h"

namespace rl::agents::alpha_zero::modules
{
    class FixedValueSupportOutput : public BaseOutput
    {
        public:
            FixedValueSupportOutput(
                const torch::Tensor &prior_logits,
                const torch::Tensor &value_logits,
                float v_min,
                float v_max,
                int n_atoms
            );

            torch::Tensor value_estimates() const override;
            torch::Tensor value_loss(const torch::Tensor &rewards) const override;
        
        private:
            float v_min, v_max;
            torch::Tensor atoms, value_logits;
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_MODULES_FIXED_SUPPORT_H_ */
