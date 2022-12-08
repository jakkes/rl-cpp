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
            BaseOutput(const torch::Tensor &policy_logits);
            
            virtual ~BaseOutput() = default;

            inline
            const rl::policies::Categorical &policy() const { return policy_; }

            inline
            torch::Tensor policy_loss(const torch::Tensor &target_policy) const {
                return (target_policy * torch::log_softmax(policy_logits, -1)).sum(-1);
            }

            virtual
            torch::Tensor value_estimates() const = 0;

            virtual
            torch::Tensor value_loss(const torch::Tensor &rewards) const = 0;
        
        private:
            const rl::policies::Categorical policy_;
            const torch::Tensor policy_logits;
    };

    class Base : public torch::nn::Module
    {
        public:
            
            virtual
            std::unique_ptr<BaseOutput> forward(const torch::Tensor &states) = 0;
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_MODULES_BASE_H_ */
