#ifndef RL_AGENTS_DQN_MODULES_BASE_H_
#define RL_AGENTS_DQN_MODULES_BASE_H_


#include <memory>

#include <torch/torch.h>

#include <rl/policies/constraints/categorical_mask.h>


namespace rl::agents::dqn::modules
{
    /**
     * @brief Base output for DQN modules.
     * 
     */
    class BaseOutput
    {
        public:

            /**
             * @brief Computes the value for every action in the given state.
             * 
             * @return torch::Tensor Value of each action, along the final dimension.
             */
            virtual torch::Tensor value() const = 0;

            /**
             * @brief Computes the value of a specific action(s).
             * 
             * @param actions Action(s)
             * @return torch::Tensor Value of the given action in the current state.
             */
            virtual torch::Tensor value(const torch::Tensor &actions) const = 0;

            /**
             * @brief Applies the given mask to the value output.
             * 
             * @param masks Action masks, boolean tensor.
             * @return torch::Tensor 
             */
            virtual void apply_mask(const rl::policies::constraints::CategoricalMask &masks) = 0;
    };

    /**
     * @brief Base class for DQN modules.
     * 
     */
    class Base : public torch::nn::Module
    {
        public:
            
            /**
             * @brief Computes Q values for the given states.
             * 
             * @param states States
             * @return std::unique_ptr<BaseOutput> Value data
             */
            virtual std::unique_ptr<BaseOutput> forward(const torch::Tensor &states) = 0;
            
            /**
             * @brief Computes the loss of the given data.
             * 
             * @param states States
             * @param masks Masks
             * @param rewards Rewards
             * @param not_terminals Not terminals
             * @param next_states Next states
             * @param next_masks Next masks
             * @param discount Discount factor
             * @return torch::Tensor Loss
             */
            virtual torch::Tensor loss(
                const torch::Tensor &states,
                const rl::policies::constraints::CategoricalMask &masks,
                const torch::Tensor &rewards,
                const torch::Tensor &not_terminals,
                const torch::Tensor &next_states,
                const rl::policies::constraints::CategoricalMask &next_masks,
                float discount
            ) = 0;
    };
}


#endif /* RL_AGENTS_DQN_MODULES_BASE_H_ */
