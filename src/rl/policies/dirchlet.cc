#include "rl/policies/dirchlet.h"


using namespace torch::indexing;

namespace rl::policies
{
    static inline
    torch::Tensor lnbeta(const torch::Tensor &alpha) {
        return torch::special::gammaln(alpha).sum(-1) - torch::special::gammaln(alpha.sum(-1));
    }

    Dirchlet::Dirchlet(torch::Tensor coefficients)
    :
    coefficients{ register_buffer("coefficients", coefficients) },
    dim{coefficients.size(-1)},
    gamma{ register_module("gamma", std::make_shared<Gamma>(coefficients, torch::ones_like(coefficients))) }
    {}

    torch::Tensor Dirchlet::sample() const
    {
        auto x = gamma->sample();
        return (x / x.sum(-1, true));
    }

    torch::Tensor Dirchlet::entropy() const
    {
        return
            lnbeta(coefficients)
            + (coefficients.sum(-1) - dim) * torch::special::digamma(coefficients.sum(-1))
            - (
                (coefficients - 1) * torch::special::digamma(coefficients)
            ).sum(-1);
    }

    torch::Tensor Dirchlet::log_prob(const torch::Tensor &x) const
    {
        assert((x < 1.0).all().item().toBool());
        assert((x > 0.0).all().item().toBool());
        return ((coefficients - 1) * x.log()).sum(-1) - lnbeta(coefficients);
    }

    torch::Tensor Dirchlet::prob(const torch::Tensor &value) const
    {
        return log_prob(value).exp_();
    }

    std::unique_ptr<Base> Dirchlet::index(const std::vector<torch::indexing::TensorIndex> &indexing) const {
        return std::make_unique<Dirchlet>(coefficients.index(indexing));
    }
}
