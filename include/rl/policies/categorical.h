#ifndef RL_POLICIES_CATEGORICAL_H_
#define RL_POLICIES_CATEGORICAL_H_


#include <torch/torch.h>

#include "base.h"


namespace rl::policies
{
    class Categorical : public Base{
        public:
            Categorical(const torch::Tensor &probabilities);
            const torch::Tensor sample() const;
            const torch::Tensor log_prob(const torch::Tensor &value) const;
            const torch::Tensor get_probabilities() const;

        private:
            torch::Tensor probabilities;
            torch::Tensor log_probabilities;
            torch::Tensor cumsummed;
            int64_t batchsize;
            bool batch;
            torch::Tensor batchvec;
    };
}


#endif /* RL_POLICIES_CATEGORICAL_H_ */
