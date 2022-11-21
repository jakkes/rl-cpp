#ifndef RL_AGENTS_ALPHA_ZERO_MODULES_BASE_H_
#define RL_AGENTS_ALPHA_ZERO_MODULES_BASE_H_


#include <memory>

#include <rl/policies/categorical.h>

namespace rl::agents::alpha_zero::modules
{
    class Base
    {
        public:
            Base(const torch::Tensor &prior_probabilities);

            inline
            const rl::policies::Categorical &priors() const { return priors_; }

            virtual torch::Tensor value_estimates() const = 0;
        
        private:
            rl::policies::Categorical priors_;
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_MODULES_BASE_H_ */
