#ifndef RL_POLICIES_CATEGORICAL_H_
#define RL_POLICIES_CATEGORICAL_H_

#include <vector>

#include <torch/torch.h>

#include "base.h"


namespace rl::policies
{
    class Categorical : public Base{
        public:
            Categorical(const torch::Tensor &probabilities);
            
            torch::Tensor sample() const;
            torch::Tensor entropy() const;
            
            torch::Tensor log_prob(const torch::Tensor &value) const;
            torch::Tensor prob(const torch::Tensor &value) const;
            
            const torch::Tensor get_probabilities() const;
            
            void include(std::shared_ptr<constraints::Base> constraint) override;

        private:
            torch::Tensor probabilities;
            torch::Tensor cumsummed;
            std::vector<int64_t> sample_shape{};
            int64_t dim;
            bool batch;
            torch::Tensor batchvec;

            void check_probabilities();
            void compute_internals();
    };
}


#endif /* RL_POLICIES_CATEGORICAL_H_ */
