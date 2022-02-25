#ifndef RL_ENV_CONSTRAINTS_BASE_H_
#define RL_ENV_CONSTRAINTS_BASE_H_

#include <memory>
#include <vector>

#include <torch/torch.h>



namespace rl::policies::constraints
{
    class Base
    {
        public:
            virtual torch::Tensor contains(const torch::Tensor &x) const = 0;
    };
}

#endif /* RL_ENV_CONSTRAINTS_BASE_H_ */
