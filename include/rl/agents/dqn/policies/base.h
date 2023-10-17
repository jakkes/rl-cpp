#ifndef RL_AGENTS_DQN_POLICIES_BASE_H_
#define RL_AGENTS_DQN_POLICIES_BASE_H_


#include <memory>

#include <rl/policies/categorical.h>

namespace rl::agents::dqn::policies
{

    /**
     * @brief DQN policies convert outputs into actual action distributions.
     */
    class Base
    {
        public:

            virtual ~Base() = default;

            /**
             * @brief Constructs a discrete action policy from the output of a DQN
             * module.
             * 
             * @param values DQN values
             * @return std::unique_ptr<rl::policies::Base> Action policy
             */
            virtual
            std::unique_ptr<rl::policies::Categorical> policy(const torch::Tensor &values) = 0;
    };
}

#endif /* RL_AGENTS_DQN_POLICIES_BASE_H_ */
