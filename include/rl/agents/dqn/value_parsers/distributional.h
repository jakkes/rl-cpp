#ifndef RL_AGENTS_DQN_VALUE_PARSERS_DISTRIBUTIONAL_H_
#define RL_AGENTS_DQN_VALUE_PARSERS_DISTRIBUTIONAL_H_


#include <torch/torch.h>

#include "base.h"


namespace rl::agents::dqn::value_parsers
{
    class Distributional : public Base
    {
        public:
            Distributional(const torch::Tensor &atoms, bool cuda_graph_compatible = true);
            ~Distributional() = default;

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

        private:
            const torch::Tensor atoms;
            const bool cuda_graph_compatible;
    };
}

#endif /* RL_AGENTS_DQN_VALUE_PARSERS_DISTRIBUTIONAL_H_ */
