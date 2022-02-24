#ifndef RL_ENV_CONSTRAINTS_CATEGORICAL_MASK_H_
#define RL_ENV_CONSTRAINTS_CATEGORICAL_MASK_H_

#include "base.h"


namespace rl::policies::constraints
{
    class CategoricalMask : public Base
    {
        public:
            CategoricalMask(const torch::Tensor &mask);

            const torch::Tensor contains(const torch::Tensor &value) const;
        private:
            torch::Tensor mask;
    };
}

#endif /* RL_ENV_CONSTRAINTS_CATEGORICAL_MASK_H_ */
