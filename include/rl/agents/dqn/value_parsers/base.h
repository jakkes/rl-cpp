#ifndef RL_AGENTS_DQN_VALUE_PARSERS_BASE_H_
#define RL_AGENTS_DQN_VALUE_PARSERS_BASE_H_


#include <torch/torch.h>


namespace rl::agents::dqn::value_parsers
{
    /**
     * @brief Base class for value parsers.
     * 
     * Value parsers parse network outputs and action masks into action values as well
     * as loss functions.
     */
    class Base
    {
        public:

            virtual ~Base() = default;

            /**
             * @brief Computes the value of each action in the given output.
             * 
             * @param module_outputs Output of the network.
             * @param masks Action mask.
             * 
             * @return torch::Tensor Value of each action, along the final dimension.
            */
            virtual torch::Tensor values(
                const torch::Tensor &module_outputs, const torch::Tensor &masks
            ) const = 0;

            /**
             * @brief Computes the loss function from a given observation.
             * 
             * @param module_outputs Output of the network.
             * @param masks Action masks.
             * @param actions Actions taken.
             * @param rewards Rewards received.
             * @param not_terminals Inverted terminal flags.
             * @param next_module_outputs Output of the network for the next state.
             * @param next_masks Action masks for the next state.
             * @param next_actions Actions taken in the next state.
             * @param discount Discount factor.
             * 
             * @return torch::Tensor Loss, per batch.
             */
            virtual torch::Tensor loss(
                const torch::Tensor &module_outputs,
                const torch::Tensor &masks,
                const torch::Tensor &actions,
                const torch::Tensor &rewards,
                const torch::Tensor &not_terminals,
                const torch::Tensor &next_module_outputs,
                const torch::Tensor &next_masks,
                const torch::Tensor &next_actions,
                float discount
            ) const = 0;
    };
}

#endif /* RL_AGENTS_DQN_VALUE_PARSERS_BASE_H_ */
