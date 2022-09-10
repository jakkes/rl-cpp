#include "rl/agents/dqn/modules/dqn.h"


namespace rl::agents::dqn::modules
{
    torch::Tensor DQNOutput::loss(
        const torch::Tensor &actions, const torch::Tensor &rewards,
        const torch::Tensor &not_terminals, const BaseOutput &next_output,
        const torch::Tensor &next_actions, float discount
    )
    {
        const DQNOutput &next_output_ = dynamic_cast<const DQNOutput&>(next_output);
        torch::Tensor batchvec = torch::arange(actions.size(0), torch::TensorOptions{}.device(actions.device()));

        auto next_values = discount * next_output_.values.index({batchvec, next_actions});
        auto next_values_actual = torch::where(not_terminals, next_values, torch::zeros_like(next_values));
        assert (!next_values_actual.requires_grad());
        assert (values.requires_grad());

        return (rewards + next_values_actual - values.index({batchvec, actions})).square_();
    }
}
