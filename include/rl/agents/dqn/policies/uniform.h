#ifndef LIBS_RL_CPP_INCLUDE_RL_AGENTS_DQN_POLICIES_UNIFORM_H_
#define LIBS_RL_CPP_INCLUDE_RL_AGENTS_DQN_POLICIES_UNIFORM_H_


#include "base.h"

namespace rl::agents::dqn::policies {

    /**
     * @brief Policy converting values into action distributions using softmax. Larger
     * values will be more likely to be chosen.
    */
    class Uniform : public Base
    {
        public:
            Uniform() = default;
            ~Uniform() = default;

            std::unique_ptr<rl::policies::Categorical> policy(
                const torch::Tensor &values,
                const torch::Tensor &masks
            ) override {
                auto probability_weights = torch::ones_like(values).where(masks, torch::zeros_like(values));
                return std::make_unique<rl::policies::Categorical>(probability_weights);
            }
    };
}

#endif /* LIBS_RL_CPP_INCLUDE_RL_AGENTS_DQN_POLICIES_UNIFORM_H_ */
