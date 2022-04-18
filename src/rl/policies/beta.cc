#include "rl/policies/beta.h"
#include "rl/policies/constraints/constraints.h"


namespace rl::policies
{

    static
    void check_sizes(torch::Tensor alpha, torch::Tensor beta, torch::Tensor a, torch::Tensor b)
    {
        if (
            alpha.sizes() != beta.sizes()
            || alpha.sizes() != a.sizes()
            || alpha.sizes() != b.sizes()
        ) {
            throw std::invalid_argument{"All shapes must be equal."};
        }
    }

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

    Beta::Beta(torch::Tensor alpha, torch::Tensor beta, torch::Tensor a, torch::Tensor b)
    :
    x{alpha, torch::ones_like(alpha)},
    y{beta, torch::ones_like(beta)},
    a{a},
    b{b},
    alpha{alpha},
    beta{beta}
    {
        check_sizes(alpha, beta, a, b);
    }

    Beta::Beta(torch::Tensor alpha, torch::Tensor beta)
    : Beta(alpha, beta, torch::zeros_like(alpha), torch::ones_like(alpha)) {}

    torch::Tensor Beta::prob(const torch::Tensor &value) const
    {
        return log_prob(value).exp_();
    }

    torch::Tensor Beta::log_prob(const torch::Tensor &value) const
    {
        auto x = invmap(value, a, b);
        return 
            (alpha - 1) * x.log()
            + (beta - 1) * (1 - x).log()
            + torch::special::gammaln(alpha + beta)
            - torch::special::gammaln(alpha)
            - torch::special::gammaln(beta);
    }

    torch::Tensor Beta::sample() const
    {
        auto X = x.sample();
        auto Y = y.sample();
        auto sample = X / (X + Y);
        return map(sample, a, b).clamp_(a+1e-6, b-1e-6);
    }

    torch::Tensor Beta::entropy() const
    {
        return 
            torch::special::gammaln(alpha + beta)
            - torch::special::gammaln(alpha)
            - torch::special::gammaln(beta)
            - (alpha - 1) * torch::special::digamma(alpha)
            - (beta - 1) * torch::special::digamma(beta)
            + (alpha + beta - 2) * torch::special::digamma(alpha + beta);
    }

    void Beta::include(std::shared_ptr<constraints::Base> constraint)
    {

        auto box = std::dynamic_pointer_cast<constraints::Box>(constraint);
        if (box)
        {
            if ((box->lower_bound() > a).any().item().toBool()) {
                throw std::invalid_argument{"Box lower bound is not fulfilled by policy."};
            }
            if ((box->upper_bound() < b).any().item().toBool()) {
                throw std::invalid_argument{"Box upper bound is not fulfilled by policy."};
            }
            return;
        }

        throw UnsupportedConstraintException{};
    }
}
