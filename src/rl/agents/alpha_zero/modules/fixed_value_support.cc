#include "rl/agents/alpha_zero/modules/fixed_value_support.h"

#include <rl/agents/utils/distributional_loss.h>


namespace rl::agents::alpha_zero::modules
{
    FixedValueSupportOutput::FixedValueSupportOutput(
        const torch::Tensor &prior_logits,
        const torch::Tensor &value_logits,
        float v_min,
        float v_max,
        int n_atoms
    ) :
        BaseOutput{prior_logits},
        value_logits{value_logits},
        v_min{v_min},
        v_max{v_max}
    {
        atoms = torch::linspace(v_min, v_max, n_atoms, value_logits.options());
    }

    torch::Tensor FixedValueSupportOutput::value_estimates() const
    {
        return (atoms * torch::softmax(value_logits, -1)).sum(-1);
    }

    torch::Tensor FixedValueSupportOutput::value_loss(const torch::Tensor &rewards) const
    {
        return rl::agents::utils::distributional_loss(
            value_logits,
            rewards,
            torch::zeros(
                {rewards.size(0)},
                torch::TensorOptions{}.dtype(torch::kBool).device(rewards.device())
            ),
            value_logits.detach(),
            atoms,
            1.0,
            false
        );
    }
}
