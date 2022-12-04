#ifndef RL_AGENTS_ALPHA_ZERO_MODULES_FIXED_SUPPORT_H_
#define RL_AGENTS_ALPHA_ZERO_MODULES_FIXED_SUPPORT_H_


#include <torch/torch.h>

#include "base.h"

namespace rl::agents::alpha_zero::modules
{
    class FixedValueSupport : public Base
    {
        public:
            FixedSupport(float v_min, float v_max, int n_atoms);
        
        private:
            const float v_min, v_max;
            const int n_atoms;

            torch::Tensor atoms;
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_MODULES_FIXED_SUPPORT_H_ */
