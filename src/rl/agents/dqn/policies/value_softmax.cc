#include "rl/agents/dqn/policies/value_softmax.h"


namespace rl::agents::dqn::policies {

    std::unique_ptr<rl::policies::Categorical> ValueSoftmax::policy(
        const torch::Tensor &values,
        const torch::Tensor &masks
    ) {
        auto masked_values = values.where(masks, torch::zeros_like(values) - INFINITY);
        auto probabilities = torch::softmax(masked_values / temperature->get(), -1);
        return std::make_unique<rl::policies::Categorical>(probabilities);
    }
}
