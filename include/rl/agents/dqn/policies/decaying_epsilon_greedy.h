#ifndef RL_CPP_INCLUDE_RL_AGENTS_DQN_POLICIES_DECAYING_EPSILON_GREEDY_H_
#define RL_CPP_INCLUDE_RL_AGENTS_DQN_POLICIES_DECAYING_EPSILON_GREEDY_H_

#include <cmath>

#include "base.h"
#include "epsilon_greedy.h"

namespace rl::agents::dqn::policies
{
    class DecayingEpsilonGreedy : public Base
    {
        public:

            DecayingEpsilonGreedy(float start_epsilon, float end_epsilon, int64_t half_steps)
            : 
            start_epsilon{start_epsilon}, end_epsilon{end_epsilon},
            half_steps{half_steps}, policy_{start_epsilon}
            {
                decay_factor = - std::log(2.0) / half_steps;
            }

            ~DecayingEpsilonGreedy() = default;

            std::unique_ptr<rl::policies::Categorical> policy(
                    const rl::agents::dqn::modules::BaseOutput &output
            ) override {
                policy_.update_epsilon( end_epsilon + (start_epsilon - end_epsilon) * std::exp(decay_factor * steps++) );
                return policy_.policy(output);
            }

        private:
            const float start_epsilon, end_epsilon;
            const int64_t half_steps;
            int64_t steps{0};
            float decay_factor;
            EpsilonGreedy policy_;
    };
}

#endif /* RL_CPP_INCLUDE_RL_AGENTS_DQN_POLICIES_DECAYING_EPSILON_GREEDY_H_ */
