#ifndef RL_POLICIES_CATEGORICAL_H_
#define RL_POLICIES_CATEGORICAL_H_

#include <vector>

#include <torch/torch.h>

#include "base.h"


namespace rl::policies
{
    /**
     * @brief Discrete policy.
     */
    class Categorical : public Base {
        public:
            /**
             * @brief Construct a new Categorical object
             * 
             * @param probabilities Probabilities of each action.
             */
            Categorical(const torch::Tensor &probabilities);

            /**
             * @brief Construct a new Categorical object
             * 
             * @param probabilities Probabilities of each action.
             * @param values Value assigned to each action.
             */
            Categorical(const torch::Tensor &probabilities, const torch::Tensor &values);
            ~Categorical() = default;
            
            torch::Tensor sample() const;
            torch::Tensor entropy() const;
            
            torch::Tensor log_prob(const torch::Tensor &value) const;
            torch::Tensor prob(const torch::Tensor &value) const;
            
            const torch::Tensor get_probabilities() const;
            
            void include(std::shared_ptr<constraints::Base> constraint) override;

        private:
            torch::Tensor probabilities, cumsummed, batchvec, values;
            const std::vector<int64_t> sample_shape;
            const int64_t dim;

            void check_probabilities();
            void compute_internals();
    };
}


#endif /* RL_POLICIES_CATEGORICAL_H_ */
