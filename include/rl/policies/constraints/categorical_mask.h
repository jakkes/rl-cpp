#ifndef RL_ENV_CONSTRAINTS_CATEGORICAL_MASK_H_
#define RL_ENV_CONSTRAINTS_CATEGORICAL_MASK_H_

#include "base.h"


namespace rl::policies::constraints
{
    class CategoricalMask : public Base
    {
        public:
            CategoricalMask(const torch::Tensor &mask);

            torch::Tensor contains(const torch::Tensor &value) const;
            std::unique_ptr<Base> stack(const std::vector<std::shared_ptr<Base>> &constraints) const;
        private:
            const torch::Tensor mask;
            const bool batch;
            const int64_t batchsize;
            torch::Tensor batchvec;
    };
}

#endif /* RL_ENV_CONSTRAINTS_CATEGORICAL_MASK_H_ */
