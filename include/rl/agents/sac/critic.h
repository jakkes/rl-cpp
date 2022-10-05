#ifndef RL_AGENTS_SAC_CRITIC_H_
#define RL_AGENTS_SAC_CRITIC_H_


#include <torch/torch.h>
#include <rl/policies/base.h>

namespace rl::agents::sac
{
    class CriticOutput
    {
        public:
            CriticOutput(torch::Tensor value)
            : value_{value}
            {}

            inline
            const torch::Tensor &value() const { return value_; }
        
        private:
            torch::Tensor value_;
    };

    class Critic : public torch::nn::Module
    {
        public:
            virtual ~Critic() = default;

            virtual
            std::unique_ptr<CriticOutput> forward(
                const torch::Tensor &states,
                const torch::Tensor &actions
            ) = 0;

            virtual std::unique_ptr<Critic> clone() const = 0;
    };
}

#endif /* RL_AGENTS_SAC_CRITIC_H_ */
