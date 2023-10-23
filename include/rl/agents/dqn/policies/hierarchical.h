#ifndef LIBS_RL_CPP_INCLUDE_RL_AGENTS_DQN_POLICIES_HIERARCHICAL_H_
#define LIBS_RL_CPP_INCLUDE_RL_AGENTS_DQN_POLICIES_HIERARCHICAL_H_


#include <algorithm>
#include <vector>
#include <memory>

#include <rl/utils/float_control/base.h>
#include <rl/utils/float_control/fixed.h>
#include "base.h"

namespace rl::agents::dqn::policies {

    /**
     * @brief Policy converting values into action distributions using softmax. Larger
     * values will be more likely to be chosen.
    */
    class Hierarchical : public Base
    {
        public:
            Hierarchical(
                const std::vector<std::shared_ptr<Base>> &policies,
                const std::vector<std::shared_ptr<rl::utils::float_control::Base>> &probabilities
            ) : policies{policies}, probabilities{probabilities} {
                if (policies.size() != probabilities.size()) {
                    throw std::invalid_argument("policies and probabilities must be the same size");
                }
            }

            Hierarchical(
                const std::vector<std::shared_ptr<Base>> &policies,
                const std::vector<float> &probabilities
            ) : policies{policies}
            {
                if (policies.size() != probabilities.size()) {
                    throw std::invalid_argument("policies and probabilities must be the same size");
                }

                for (auto probability : probabilities) {
                    this->probabilities.push_back(
                        std::make_shared<rl::utils::float_control::Fixed>(probability)
                    );
                }
            }

            ~Hierarchical() = default;

            std::unique_ptr<rl::policies::Categorical> policy(
                const torch::Tensor &values,
                const torch::Tensor &masks
            ) override;

        private:
            std::vector<std::shared_ptr<Base>> policies;
            std::vector<std::shared_ptr<rl::utils::float_control::Base>> probabilities;
    };
}

#endif /* LIBS_RL_CPP_INCLUDE_RL_AGENTS_DQN_POLICIES_HIERARCHICAL_H_ */
