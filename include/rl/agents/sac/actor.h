#ifndef RL_AGENTS_SAC_ACTOR_H_
#define RL_AGENTS_SAC_ACTOR_H_


#include <torch/torch.h>

#include <rl/agents/utils/distributional_value.h>


namespace rl::agents::sac
{
    class ActorOutput
    {
        public:
            ActorOutput(
                const torch::Tensor &mean,
                const torch::Tensor &std,
                const rl::agents::utils::DistributionalValue &value
            )
            : mean_{mean}, std_{std}, value_{value}
            {}

            inline
            auto mean() const { return mean_; }

            inline
            auto std() const { return std_; }

            inline
            const auto &value() const { return value_; }

            inline
            auto sample() const {
                return mean_ + torch::randn_like(std_.detach()) * std_;
            }
        
        private:
            torch::Tensor mean_, std_;
            rl::agents::utils::DistributionalValue value_;
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
