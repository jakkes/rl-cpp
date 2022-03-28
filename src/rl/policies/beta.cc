#include "rl/policies/beta.h"

#include <cmath>


#include "rl/policies/constraints/constraints.h"


#define RL_BETA_RESOLUTION 1000
using namespace torch::indexing;


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

        x = torch::linspace(0.0, 1.0, RL_BETA_RESOLUTION + 2, flat_alpha.options()).index({Slice(1, -1)});

        pdf = x.pow(alpha.unsqueeze(-1) - 1) * (1 - x).pow(beta.unsqueeze(-1) - 1);
        pdf = pdf / pdf.sum(-1, true);
        cdf = pdf.cumsum(-1);

        for (int64_t i = 0; i < flat_alpha.size(0); i++) {
            flat_c.index_put_({i}, std::beta(flat_alpha.index({i}).item().toFloat(), flat_beta.index({i}).item().toFloat()));
        }

        c = flat_c.view_as(a);

        sample_shape.reserve(cdf.sizes().size());
        sample_shape.clear();
        int i = 0;
        int64_t batchsize = 1;
        for (; i < cdf.sizes().size() - 1; i++) {
            batchsize *= cdf.size(i);
            sample_shape.push_back(cdf.size(i));
        }
        sample_shape.push_back(1);
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
        auto indices = (torch::rand(sample_shape, cdf.options()) > cdf).sum(-1);
        return map(
            torch::ones_like(indices) * (indices + 1) / (RL_BETA_RESOLUTION + 1),
            a,
            b
        );
    }

    torch::Tensor Beta::entropy() const
    {
        return - (x.log() * pdf / (RL_BETA_RESOLUTION + 1)).sum(-1);
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
