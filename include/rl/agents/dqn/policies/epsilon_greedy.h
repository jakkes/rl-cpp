#ifndef RL_AGENTS_DQN_POLICIES_EPSILON_GREEDY_H_
#define RL_AGENTS_DQN_POLICIES_EPSILON_GREEDY_H_


#include "base.h"
#include <rl/utils/float_control/base.h>
#include <rl/utils/float_control/fixed.h>

namespace rl::agents::dqn::policies
{
    class EpsilonGreedy : public Base
    {
        public:

            EpsilonGreedy(std::shared_ptr<rl::utils::float_control::Base> epsilon)
            : epsilon{epsilon} {}

            EpsilonGreedy(float epsilon)
            : EpsilonGreedy(std::make_shared<rl::utils::float_control::Fixed>(epsilon))
            {}

            ~EpsilonGreedy() = default;

            std::unique_ptr<rl::policies::Categorical> policy(
                    const rl::agents::dqn::modules::BaseOutput &output) override;

        private:
            std::shared_ptr<rl::utils::float_control::Base> epsilon;
    };
}

#endif /* RL_AGENTS_DQN_POLICIES_EPSILON_GREEDY_H_ */
