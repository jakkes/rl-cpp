#ifndef RL_ENV_CONSTRAINTS_BASE_H_
#define RL_ENV_CONSTRAINTS_BASE_H_

#include <torch/torch.h>


namespace rl::policies::constraints
{
    class Base
    {
        public:
            virtual const torch::Tensor contains(torch::Tensor x) const = 0;
    };
}

#endif /* RL_ENV_CONSTRAINTS_BASE_H_ */
