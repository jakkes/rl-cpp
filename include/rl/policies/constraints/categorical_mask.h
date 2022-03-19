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
            torch::Tensor contains(const torch::Tensor &value) const;
            std::function<std::unique_ptr<Base>(const std::vector<std::shared_ptr<Base>>&)>stack_fn() const;

            inline const torch::Tensor mask() const { return _mask; }

            friend std::unique_ptr<CategoricalMask> stack<CategoricalMask>(const std::vector<std::shared_ptr<CategoricalMask>> &constraints);
        private:
            torch::Tensor _mask{};
            torch::Tensor batchvec{};
            bool batch{};
            int64_t dim{};
            int64_t batchsize{0};
    };

    template<>
    std::unique_ptr<CategoricalMask> stack<CategoricalMask>(const std::vector<std::shared_ptr<CategoricalMask>> &constraints);
}

#endif /* RL_ENV_CONSTRAINTS_CATEGORICAL_MASK_H_ */
