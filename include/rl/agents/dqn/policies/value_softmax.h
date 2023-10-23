#ifndef LIBS_RL_CPP_INCLUDE_RL_AGENTS_DQN_POLICIES_VALUE_SOFTMAX_H_
#define LIBS_RL_CPP_INCLUDE_RL_AGENTS_DQN_POLICIES_VALUE_SOFTMAX_H_

#include <memory>
#include <rl/utils/float_control/base.h>
#include <rl/utils/float_control/fixed.h>

#include "base.h"

namespace rl::agents::dqn::policies {

    /**
     * @brief Policy converting values into action distributions using softmax. Larger
     * values will be more likely to be chosen.
    */
    class ValueSoftmax : public Base
    {
        public:
            ValueSoftmax(std::shared_ptr<rl::utils::float_control::Base> temperature) : temperature(temperature) {}
            ValueSoftmax(float temperature) : ValueSoftmax(std::make_shared<rl::utils::float_control::Fixed>(temperature)) {}
            ~ValueSoftmax() = default;

            std::unique_ptr<rl::policies::Categorical> policy(
                const torch::Tensor &values,
                const torch::Tensor &masks
            ) override;

        private:
            std::shared_ptr<rl::utils::float_control::Base> temperature;
    };
}

#endif /* LIBS_RL_CPP_INCLUDE_RL_AGENTS_DQN_POLICIES_VALUE_SOFTMAX_H_ */
