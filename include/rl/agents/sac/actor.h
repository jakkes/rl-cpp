#ifndef RL_AGENTS_SAC_ACTOR_H_
#define RL_AGENTS_SAC_ACTOR_H_


#include <torch/torch.h>

namespace rl::agents::sac
{
    class ActorOutput
    {
        public:
            ActorOutput(const torch::Tensor &mean, const torch::Tensor &std, const torch::Tensor &value)
            : mean_{mean}, std_{std}, value_{value}
            {}

            inline
            const torch::Tensor mean() const { return mean_; }

            inline
            const torch::Tensor std() const { return std_; }

            inline
            const torch::Tensor value() const { return value_; }

            inline
            torch::Tensor sample() const {
                return mean_ + torch::randn_like(std_.detach()) * std_;
            }
        
        private:
            torch::Tensor value_, mean_, std_;
    };

    class Actor : public torch::nn::Module
    {
        public:
            virtual ~Actor() = default;

            virtual
            ActorOutput forward(
                const torch::Tensor &states
            ) = 0;

            virtual std::unique_ptr<Actor> clone() const = 0;
    };
}

#endif /* RL_AGENTS_SAC_ACTOR_H_ */
