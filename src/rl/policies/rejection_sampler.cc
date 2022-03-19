#include "rl/policies/rejection_sampler.h"


namespace rl::policies
{
    RejectionSampler::RejectionSampler(std::unique_ptr<Base> distribution,
                                                const constraints::Concat &constraints)
    : distribution{std::move(distribution)}, constraints{constraints} 
    {}

    RejectionSampler::RejectionSampler(std::unique_ptr<Base> distribution,
    std::shared_ptr<constraints::Base> constraint)
    : RejectionSampler(std::move(distribution), constraints::Concat{ {constraint }})
    {}

    void RejectionSampler::include(std::shared_ptr<constraints::Base> constraint) {
        constraints.push_back(constraint);
    }

    torch::Tensor RejectionSampler::log_prob(const torch::Tensor &value) const {
        return distribution->log_prob(value);
    }

    torch::Tensor RejectionSampler::prob(const torch::Tensor &value) const {
        return distribution->prob(value);
    }

    torch::Tensor RejectionSampler::sample() const
    {
        auto sample = distribution->sample();
        auto constraints_fulfilled = constraints.contains(sample);

        while (!constraints.contains(sample).all().item().toBool()) {
            sample = distribution->sample();
        }
        return sample;
    }
}
