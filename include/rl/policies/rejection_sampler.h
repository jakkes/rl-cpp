#ifndef RL_POLICIES_REJECTION_SAMPLER_H_
#define RL_POLICIES_REJECTION_SAMPLER_H_

#include <memory>
#include <vector>

#include <torch/torch.h>

#include "base.h"
#include "constraints/cat.h"


namespace rl::policies
{

    class RejectionSampler : public Base
    {
        public:
            RejectionSampler(
                std::unique_ptr<Base> distribution,
                std::shared_ptr<constraints::Base> constraint
            );
            RejectionSampler(
                std::unique_ptr<Base> distribution,
                const constraints::Concat &constraints
            );

            void include(std::shared_ptr<constraints::Base> constraint) override;

            torch::Tensor sample() const;
            torch::Tensor log_prob(const torch::Tensor &value) const;
        
        private:
            std::unique_ptr<Base> distribution;
            constraints::Concat constraints;
    };
}

#endif /* RL_POLICIES_REJECTION_SAMPLER_H_ */
