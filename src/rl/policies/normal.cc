#include "rl/policies/normal.h"

#include "rl/policies/constraints/constraints.h"

#define M_1_SQRT_2PI 0.3989422804f  // 1 / sqrt(2pi)
#define M_LN_2PI 1.83787706641f     // ln (2pi)


namespace rl::policies
{
    Normal::Normal(const torch::Tensor &mean, const torch::Tensor &std)
    :
    mean{mean}, std{std},
    lower_bound{torch::zeros_like(mean) - INFINITY},
    upper_bound{torch::zeros_like(mean) + INFINITY}
    {
        compute_cdf_at_bounds();
    }

    inline torch::Tensor Normal::standardize(const torch::Tensor &x) const
    {
        return x.sub(mean).div_(std);
    }

    torch::Tensor Normal::sample() const
    {
        auto out = torch::randn_like(mean) * std + mean;
        auto invalid_mask = (out < lower_bound).logical_or_(out > upper_bound);

        while (invalid_mask.any().item().toBool()) {
            out.index_put_({invalid_mask}, torch::randn({invalid_mask.sum().item().toLong()}, out.options()) * std + mean);
            invalid_mask = (out < lower_bound).logical_or_(out > upper_bound);
        }

        return out;
    }

    torch::Tensor Normal::entropy() const
    {
        return 0.5 * (1 + (2 * std.square() * M_PI).log_());
    }

    torch::Tensor Normal::prob(const torch::Tensor &x) const
    {
        auto core = M_1_SQRT_2PI * torch::exp_(standardize(x).square_().div_(2));
        return core / (cdf_upper_bound - cdf_lower_bound);
    }

    torch::Tensor Normal::log_prob(const torch::Tensor &x) const
    {
        return prob(x).log_();
    }

    void Normal::include(std::shared_ptr<constraints::Base> constraint)
    {
        auto box = std::dynamic_pointer_cast<constraints::Box>(constraint);
        
        if (box)
        {
            if (box->n_action_dims() != 0) {
                throw std::invalid_argument{"Normal only supports univariate constraints."};
            }
            lower_bound = torch::max(lower_bound, box->lower_bound());
            upper_bound = torch::min(upper_bound, box->upper_bound());
            compute_cdf_at_bounds();
            return;
        }

        throw UnsupportedConstraintException{};
    }

    void Normal::compute_cdf_at_bounds()
    {
        cdf_lower_bound = 0.5 * (1 + torch::erf(M_SQRT1_2 * standardize(lower_bound)));
        cdf_upper_bound = 0.5 * (1 + torch::erf(M_SQRT1_2 * standardize(upper_bound)));
    }
}
