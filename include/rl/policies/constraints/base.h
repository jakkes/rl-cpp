#ifndef RL_ENV_CONSTRAINTS_BASE_H_
#define RL_ENV_CONSTRAINTS_BASE_H_

#include <memory>
#include <vector>
#include <functional>

#include <torch/torch.h>



namespace rl::policies::constraints
{
    class Base
    {
        public:
            virtual torch::Tensor contains(const torch::Tensor &x) const = 0;
            virtual std::function<std::unique_ptr<Base>(const std::vector<std::shared_ptr<Base>>&)> stack_fn() const = 0;
    };
}

#endif /* RL_ENV_CONSTRAINTS_BASE_H_ */
