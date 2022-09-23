#ifndef RL_AGENTS_DQN_POLICIES_EPSILON_GREEDY_H_
#define RL_AGENTS_DQN_POLICIES_EPSILON_GREEDY_H_


#include "base.h"

namespace rl::agents::dqn::policies
{
    class EpsilonGreedy : public Base
    {
        public:

            EpsilonGreedy(float epsilon) : epsilon{epsilon} {}

            ~EpsilonGreedy() = default;

            std::unique_ptr<rl::policies::Categorical> policy(
                    const rl::agents::dqn::modules::BaseOutput &output) override;

            inline
            void update_epsilon(float epsilon) { this->epsilon = epsilon; }

        private:
            float epsilon;
    };
}

#endif /* RL_AGENTS_DQN_POLICIES_EPSILON_GREEDY_H_ */
