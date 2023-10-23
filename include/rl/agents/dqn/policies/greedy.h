#ifndef LIBS_RL_CPP_INCLUDE_RL_AGENTS_DQN_POLICIES_GREEDY_H_
#define LIBS_RL_CPP_INCLUDE_RL_AGENTS_DQN_POLICIES_GREEDY_H_


#include "base.h"

namespace rl::agents::dqn::policies {

    /**
     * @brief Policy converting values into action distributions using softmax. Larger
     * values will be more likely to be chosen.
    */
    class Greedy : public Base
    {
        public:
            Greedy() = default;
            ~Greedy() = default;

            std::unique_ptr<rl::policies::Categorical> policy(
                const torch::Tensor &values,
                const torch::Tensor &masks
            ) override {
                auto masked_values = values.where(masks, torch::zeros_like(values) - INFINITY);
                auto greedy_actions = masked_values.argmax(-1);
                auto batchvec = torch::arange(greedy_actions.size(0), greedy_actions.options());
                auto probabilities = torch::zeros_like(values);
                probabilities.index_put_({batchvec, greedy_actions}, 1.0f);

                return std::make_unique<rl::policies::Categorical>(probabilities);
            }
    };
}

#endif /* LIBS_RL_CPP_INCLUDE_RL_AGENTS_DQN_POLICIES_GREEDY_H_ */
