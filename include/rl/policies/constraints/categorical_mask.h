#ifndef RL_ENV_CONSTRAINTS_CATEGORICAL_MASK_H_
#define RL_ENV_CONSTRAINTS_CATEGORICAL_MASK_H_

#include "base.h"
#include "stack.h"


namespace rl::policies::constraints
{
    class CategoricalMask : public Base
    {
        public:
            CategoricalMask(const torch::Tensor &mask);
            torch::Tensor contains(const torch::Tensor &value) const override;
            std::unique_ptr<Base> index(const std::vector<torch::indexing::TensorIndex> &indexing) const override;
            std::unique_ptr<Base> stack(const std::vector<std::shared_ptr<Base>> &constraints) const override;
            std::unique_ptr<CategoricalMask> stack(const std::vector<std::shared_ptr<CategoricalMask>> &constraints) const;

            inline const torch::Tensor mask() const { return _mask; }
        private:
            const torch::Tensor _mask{};
            const int64_t dim{};
    };
}

#endif /* RL_ENV_CONSTRAINTS_CATEGORICAL_MASK_H_ */
