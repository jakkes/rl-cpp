#ifndef RL_AGENTS_ALPHA_ZERO_MODULES_BASE_H_
#define RL_AGENTS_ALPHA_ZERO_MODULES_BASE_H_


#include <memory>

#include <torch/torch.h>

#include <rl/policies/categorical.h>

namespace rl::agents::alpha_zero::modules
{
    class BaseOutput
    {
        public:
            virtual ~BaseOutput() = default;

            virtual
            const rl::policies::Categorical &policy() const = 0;

            virtual
            torch::Tensor value_estimates() const = 0;

            virtual
            torch::Tensor value_loss(const torch::Tensor &rewards) const = 0;
    };

    class Base : public torch::nn::Module
    {
        public:
            
            virtual
            std::unique_ptr<BaseOutput> forward(const torch::Tensor &states) = 0;

            virtual
            std::unique_ptr<Base> clone() const = 0;
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_MODULES_BASE_H_ */
