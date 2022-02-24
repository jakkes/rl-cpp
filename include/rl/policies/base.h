#ifndef RL_POLICIES_BASE_H_
#define RL_POLICIES_BASE_H_

#include <torch/torch.h>


namespace rl::policies
{
    class Base{
        public:
            virtual const torch::Tensor sample() const = 0;
            virtual const torch::Tensor log_prob(const torch::Tensor &value) const = 0;
    };
}

#endif /* RL_POLICIES_BASE_H_ */
