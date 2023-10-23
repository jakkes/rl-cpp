#include "rl/agents/dqn/policies/hierarchical.h"


namespace rl::agents::dqn::policies {

    std::unique_ptr<rl::policies::Categorical> Hierarchical::policy(
        const torch::Tensor &values,
        const torch::Tensor &masks
    ) {
        std::vector<torch::Tensor> sub_policy_probabilities{}; sub_policy_probabilities.resize(policies.size());
        for (int i = 0; i < policies.size(); i++) {
            sub_policy_probabilities[i] = policies[i]->policy(values, masks)->get_probabilities();
        }

        std::vector<float> probabilities{}; probabilities.resize(policies.size());
        float sum{0.0f};
        for (int i = 0; i < policies.size(); i++) {
            probabilities[i] = this->probabilities[i]->get();
            sum += probabilities[i];
        }
        for (int i = 0; i < policies.size(); i++) {
            probabilities[i] /= sum;
        }

        return std::make_unique<rl::policies::Categorical>(
            torch::sum(
                torch::stack(sub_policy_probabilities, 1) *
                torch::tensor(probabilities).view({1, -1, 1}),
                1
            )
        );
    }
}
