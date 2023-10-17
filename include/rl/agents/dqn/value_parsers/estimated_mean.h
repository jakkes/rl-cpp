#ifndef RL_AGENTS_DQN_VALUE_PARSERS_BASIC_H_
#define RL_AGENTS_DQN_VALUE_PARSERS_BASIC_H_


#include "base.h"


namespace rl::agents::dqn::value_parsers
{
    /**
     * @brief Basic value parser for networks that output a single value per action.
    */
    class EstimatedMean : public Base
    {
        public:
            EstimatedMean() = default;
            ~EstimatedMean() = default;

            torch::Tensor values(
                const torch::Tensor &module_output, const torch::Tensor &mask
            ) const override;

            torch::Tensor loss(
                const torch::Tensor &module_output,
                const torch::Tensor &masks,
                const torch::Tensor &actions,
                const torch::Tensor &rewards,
                const torch::Tensor &not_terminals,
                const torch::Tensor &next_module_output,
                const torch::Tensor &next_masks,
                const torch::Tensor &next_actions,
                float discount
            ) const override;
    };
}

#endif /* RL_AGENTS_DQN_VALUE_PARSERS_BASIC_H_ */
