#include "rl/policies/factorized.h"


using namespace torch::indexing;

namespace rl::policies
{
    Factorized::Factorized(const std::vector<std::shared_ptr<Base>> &policies)
    : policies{policies} {
        assert(policies.size() > 0);
    }

    torch::Tensor Factorized::sample() const
    {
        std::vector<torch::Tensor> samples{};
        samples.reserve(policies.size());

        for (const auto &policy : policies) {
            samples.push_back(policy->sample());
        }

        return torch::stack(samples, -1);
    }

    torch::Tensor Factorized::entropy() const
    {
        std::vector<torch::Tensor> entropies{};
        entropies.reserve(policies.size());

        for (const auto &policy : policies) entropies.push_back(policy->entropy());

        return torch::stack(entropies, 0).sum(0);
    }

    torch::Tensor Factorized::prob(const torch::Tensor &x) const
    {
        auto out = policies[0]->prob(x.index({"...", 0}));

        for (int i = 1; i < x.size(-1); i++) {
            out.mul_(policies[i]->prob(x.index({"...", i})));
        }

        return out;
    }

    torch::Tensor Factorized::log_prob(const torch::Tensor &x) const
    {
        auto out = policies[0]->log_prob(x.index({"...", 0}));

        for (int i = 1; i < x.size(-1); i++) {
            out.add_(policies[i]->log_prob(x.index({"...", i})));
        }

        return out;
    }
}
