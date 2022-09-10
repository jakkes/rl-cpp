#include "rl/agents/dqn/policies/epsilon_greedy.h"


namespace rl::agents::dqn::policies
{
    std::unique_ptr<rl::policies::Categorical> EpsilonGreedy::policy(
                            const rl::agents::dqn::modules::BaseOutput &output) const
    {
        auto value = output.value().detach();
        auto probabilities = epsilon * torch::ones_like(value);
        auto greedy_actions = value.argmax(-1);

        auto batchvec = torch::arange(greedy_actions.size(0), greedy_actions.options());
        probabilities.index_put_({batchvec, greedy_actions}, epsilon + 1.0f);

        // Normalization taken care by policy constructor.
        return std::make_unique<rl::policies::Categorical>(probabilities);
    }
}
