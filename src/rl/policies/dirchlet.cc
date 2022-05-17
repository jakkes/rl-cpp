#include "rl/policies/dirchlet.h"


using namespace torch::indexing;

namespace rl::policies
{

    static inline
    torch::Tensor map(const torch::Tensor &x, const torch::Tensor &a, const torch::Tensor &b)
    {
        return a + (b - a) * x;
    }

    static inline
    torch::Tensor invmap(const torch::Tensor &x, const torch::Tensor &a, const torch::Tensor &b)
    {
        return (x - a) / (b - a);
    }

    static inline
    torch::Tensor lnbeta(const torch::Tensor &alpha) {
        return torch::special::gammaln(alpha).sum(-1) - torch::special::gammaln(alpha.sum(-1));
    }

    Dirchlet::Dirchlet(torch::Tensor coefficients)
    : Dirchlet{coefficients, torch::zeros_like(coefficients), torch::ones_like(coefficients)}
    {}

    Dirchlet::Dirchlet(torch::Tensor coefficients, torch::Tensor a, torch::Tensor b)
    :
    coefficients{coefficients},
    a{a},
    b{b},
    dim{coefficients.size(-1)},
    gamma{coefficients, torch::ones_like(coefficients)}
    {}

    torch::Tensor Dirchlet::sample() const
    {
        auto x = gamma.sample();
        auto out = x / x.sum(-1, true);
        return map(out, a, b);
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

    torch::Tensor Dirchlet::log_prob(const torch::Tensor &value) const
    {
        auto x = invmap(value, a, b);
        return ((coefficients - 1) * x.log_()).sum(-1) - lnbeta(coefficients);
    }

    torch::Tensor Dirchlet::prob(const torch::Tensor &value) const
    {
        return log_prob(value).exp_();
    }

    std::unique_ptr<Base> Dirchlet::index(const std::vector<torch::indexing::TensorIndex> &indexing) const {
        return std::make_unique<Dirchlet>(coefficients.index(indexing), a.index(indexing), b.index(indexing));
    }
}
