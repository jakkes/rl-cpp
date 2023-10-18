#include "rl/agents/dqn/policies/epsilon_greedy.h"


namespace rl::agents::dqn::policies
{
    std::unique_ptr<rl::policies::Categorical> EpsilonGreedy::policy(
        const torch::Tensor &values,
        const torch::Tensor &masks
    )
    {
        auto epsilon = this->epsilon->get();

        auto masked_values = values.where(masks, torch::zeros_like(values) - INFINITY);
        auto greedy_actions = masked_values.argmax(-1);

        auto base_prob = epsilon / masks.sum(-1).item().toLong();
        auto probabilities = base_prob * torch::where(
            masks, torch::ones_like(values), torch::zeros_like(values)
        );

        auto batchvec = torch::arange(greedy_actions.size(0), greedy_actions.options());
        probabilities.index_put_({batchvec, greedy_actions}, 1.0f - epsilon + base_prob);

        return std::make_unique<rl::policies::Categorical>(probabilities);
    }
}
