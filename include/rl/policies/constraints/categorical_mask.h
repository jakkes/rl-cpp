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

            inline const torch::Tensor mask() const { return _mask; }

            friend std::unique_ptr<CategoricalMask> __stack_impl<CategoricalMask>(const std::vector<std::shared_ptr<CategoricalMask>> &constraints);
        private:
            const torch::Tensor _mask{};
            const int64_t dim{};
    };

    template<>
    std::unique_ptr<CategoricalMask> __stack_impl<CategoricalMask>(const std::vector<std::shared_ptr<CategoricalMask>> &constraints);
}

#endif /* RL_ENV_CONSTRAINTS_CATEGORICAL_MASK_H_ */
