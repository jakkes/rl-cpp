#include "rl/agents/dqn/value_parsers/estimated_mean.h"


namespace rl::agents::dqn::value_parsers
{
    torch::Tensor EstimatedMean::values(
        const torch::Tensor &module_output, const torch::Tensor &mask
    ) const
    {
        return module_output.where(mask, torch::zeros_like(module_output.detach()) - INFINITY);
    }

    torch::Tensor EstimatedMean::loss(
        const torch::Tensor &module_output,
        const torch::Tensor &masks,
        const torch::Tensor &actions,
        const torch::Tensor &rewards,
        const torch::Tensor &not_terminals,
        const torch::Tensor &next_module_output,
        const torch::Tensor &next_masks,
        const torch::Tensor &next_actions,
        float discount
    ) const
    {
        auto batchvec = torch::arange(
            actions.size(0), torch::TensorOptions{}.device(actions.device())
        );
        auto values = (
            module_output
            .where(masks, torch::zeros_like(module_output.detach()) - INFINITY)
            .index({batchvec, actions})
        );
        auto next_values = (
            discount
                * next_module_output.where(
                    next_masks,
                    torch::zeros_like(next_module_output.detach()) - INFINITY
                )
                .index({batchvec, next_actions})
        );
        assert (!next_values.requires_grad());
        assert (!rewards.requires_grad());
        assert (values.requires_grad());

        return (
            rewards 
            + next_values.where(not_terminals, torch::zeros_like(next_values)) - values
        ).square_();
    }
}
