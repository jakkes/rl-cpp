#include "rl/agents/dqn/policies/epsilon_greedy.h"


namespace rl::agents::dqn::policies
{
    std::unique_ptr<rl::policies::Categorical> EpsilonGreedy::policy(const torch::Tensor &values)
    {
        auto epsilon = this->epsilon->get();

        float base_prob = epsilon / values.size(-1);
        auto probabilities = base_prob * torch::ones_like(values);
        auto greedy_actions = values.argmax(-1);

        auto batchvec = torch::arange(greedy_actions.size(0), greedy_actions.options());
        probabilities.index_put_({batchvec, greedy_actions}, 1.0f + base_prob - epsilon);

        // Normalization taken care by policy constructor.
        return std::make_unique<rl::policies::Categorical>(probabilities);
    }
}
