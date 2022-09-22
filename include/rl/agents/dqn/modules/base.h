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

            virtual ~BaseOutput() = default;

            /**
             * @brief Computes the value for every action in the given state.
             * 
             * @return torch::Tensor Value of each action, along the final dimension.
             */
            virtual const torch::Tensor value() const = 0;

            /**
             * @brief Computes the greedy action of the output.
             * 
             * @return torch::Tensor greedy actions
             */
            inline torch::Tensor greedy_action() const {
                return value().argmax(-1);
            }

            /**
             * @brief Applies the given mask to the value output.
             * 
             * @param masks Action masks, boolean tensor.
             * @return torch::Tensor 
             */
            virtual void apply_mask(const torch::Tensor &mask) = 0;

            /**
             * @brief Computes the loss of the output when applied to the given rewards,
             * terminal flags, output of the next states, and specified next actions.
             * 
             * @param actions actions
             * @param rewards rewards
             * @param not_terminals inverted terminal flags
             * @param next_output output of next states
             * @param next_actions next actions
             * @param discount discount factor
             * @return torch::Tensor 
             */
            virtual torch::Tensor loss(
                const torch::Tensor &actions,
                const torch::Tensor &rewards,
                const torch::Tensor &not_terminals,
                const BaseOutput &next_output,
                const torch::Tensor &next_actions,
                float discount
            ) = 0;


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
             * @brief Clones the module.
             * 
             * @return std::unique_ptr<Base> Cloned module
             */
            virtual std::unique_ptr<Base> clone() const = 0;
    };
}


#endif /* RL_AGENTS_DQN_MODULES_BASE_H_ */
