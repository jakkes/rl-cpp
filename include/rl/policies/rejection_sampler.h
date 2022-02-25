#ifndef RL_POLICIES_REJECTION_SAMPLER_H_
#define RL_POLICIES_REJECTION_SAMPLER_H_

#include <memory>
#include <vector>

#include <torch/torch.h>

#include "base.h"
#include "constraints/cat.h"


namespace rl::policies
{
    namespace constraints {
        class Base;
    }

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

            void include(const constraints::Base &constraint) override;

            const torch::Tensor sample() const;
            const torch::Tensor log_prob(const torch::Tensor &value) const;
            std::unique_ptr<Base> clone() const;
        
        private:
            std::unique_ptr<Base> distribution;
            constraints::Concat constraints;
    };
}

#endif /* RL_POLICIES_REJECTION_SAMPLER_H_ */
