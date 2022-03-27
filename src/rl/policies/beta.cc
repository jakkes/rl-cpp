#include "rl/policies/beta.h"

#include <cmath>


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
    : alpha{alpha}, beta{beta}, a{a}, b{b}
    {
        auto flat_alpha = alpha.view({-1});
        auto flat_beta = beta.view({-1});
        auto flat_a = a.view({-1});
        auto flat_b = b.view({-1});
        auto flat_c = torch::empty_like(alpha.view({-1}));

        for (int64_t i = 0; i < flat_alpha.size(0); i++) {
            flat_c.index_put_({i}, std::beta(flat_alpha.index({i}).item().toFloat(), flat_beta.index({i}).item().toFloat()));
        }

        c = flat_c.view_as(a);
    }

    Beta::Beta(torch::Tensor alpha, torch::Tensor beta)
    : Beta(alpha, beta, torch::zeros_like(alpha), torch::ones_like(alpha)) {}

    torch::Tensor Beta::prob(const torch::Tensor &value) const
    {
        auto x = invmap(value, a, b);
        return x.pow(alpha-1) * (1 - x).pow_(beta-1) / c;
    }

    torch::Tensor Beta::log_prob(const torch::Tensor &value) const
    {
        auto x = invmap(value, a, b);
        return (alpha - 1) * x.log() + (beta - 1) * (1 - x).log() - c.log();
    }

    torch::Tensor Beta::sample() const
    {
        throw std::runtime_error{"Not implemented."};
    }

    torch::Tensor Beta::entropy() const
    {
        throw std::runtime_error{"Not implemented."};
    }

    void Beta::include(std::shared_ptr<constraints::Base> constraint)
    {
        throw std::runtime_error{"Not implemented."};
    }
}
