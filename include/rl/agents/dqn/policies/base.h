#ifndef RL_AGENTS_DQN_POLICIES_BASE_H_
#define RL_AGENTS_DQN_POLICIES_BASE_H_


#include <torch/torch.h>

namespace rl::agents::dqn::policies
{

    /**
     * @brief A DQN policy computes actions from Q-values.
     * 
     */
    class Base
    {
        public:

            /**
             * @brief Samples an action from the policy.
             * 
             * @return torch::Tensor action(s)
             */
            virtual torch::Tensor sample() const;
    };
}

#endif /* RL_AGENTS_DQN_POLICIES_BASE_H_ */
