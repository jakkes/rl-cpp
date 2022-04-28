#include "rl/policies/gamma.h"

#include <cmath>


namespace rl::policies
{

    static inline
    torch::Tensor is_alpha_boosted(const torch::Tensor &alpha) {
        return alpha < 1.0;
    }

    static inline
    torch::Tensor boost_alpha(const torch::Tensor &alpha) {
        return torch::where(is_alpha_boosted(alpha), alpha + 1, alpha);
    }

    static inline
    torch::Tensor d(const torch::Tensor &alpha) {
        return alpha - 1.0 / 3.0;
    }

    static inline
    torch::Tensor c(const torch::Tensor &alpha) {
        return 1.0 / 3.0 / d(alpha).sqrt();
    }

    Gamma::Gamma(const torch::Tensor &alpha, const torch::Tensor &scale)
    :
    _alpha{boost_alpha(alpha.view({-1}))},
    _is_alpha_boosted{is_alpha_boosted(alpha.view({-1}))},
    _scale{scale.view({-1})},
    _d{d(boost_alpha(alpha.view({-1})))},
    _c{c(boost_alpha(alpha.view({-1})))},
    shape{alpha.sizes().begin(), alpha.sizes().end()}
    {
        if (_alpha.size(0) != _scale.size(0)) {
            throw std::invalid_argument{"Alpha and scale must be of the same size."};
        }
        if (!_alpha.gt(0).all().item().toBool()) {
            throw std::invalid_argument{"Alpha must be greater than zero."};
        }
        if (!_scale.gt(0).all().item().toBool()) {
            throw std::invalid_argument{"Scale must be greater than zero."};
        }
    }

    torch::Tensor Gamma::sample() const
    {
        auto out = torch::empty_like(_alpha);
        auto out_not_done = torch::ones_like(_alpha);

        while (out_not_done.any().item().toBool())
        {
            auto n = out_not_done.sum().item().toLong();
            auto not_done_indices = torch::where(out_not_done)[0];

            auto v = torch::zeros({n}, _alpha.options());
            auto x = torch::zeros({n}, _alpha.options());

            torch::Tensor mask;
            while( (mask = v <= 0).any().item().toBool() )
            {
                int m = mask.sum().item().toLong();
                x.index_put_(
                    {mask},
                    torch::randn({m}, x.options())
                );
                v.index_put_(
                    {mask},
                    (1 + _c.index({not_done_indices}).index({mask}) * x.index({mask})).pow_(3)
                );
            }

            auto u = torch::rand_like(v);

            auto m1 = u < 1 - 0.331 * x.pow(4);
            auto invm1 = ~m1;
            out.index_put_(
                {not_done_indices.index({m1})},
                _d.index({not_done_indices}).index({m1}) * v.index({m1})
            );
            out_not_done.index_put_(
                {not_done_indices.index({m1})},
                false
            );

            auto m2 = torch::zeros_like(m1);
            m2.index_put_(
                {invm1},
                u.index({invm1}).log() < 0.5 * x.index({invm1}).pow(2) + _d.index({not_done_indices}).index({invm1}) * (1 - v.index({invm1}) + v.index({invm1}).log())
            );
            out.index_put_(
                {not_done_indices.index({m2})},
                _d.index({not_done_indices}).index({m2}) * v.index({m2})
            );
            out_not_done.index_put_(
                {not_done_indices.index({m2})},
                false
            );
        }

        if (_is_alpha_boosted.any().item().toBool()) {
            auto uniform = torch::rand({_is_alpha_boosted.sum().item().toLong()}, out.options());
            uniform.pow_(1.0 / (_alpha.index({_is_alpha_boosted}) - 1.0));
            out.index_put_(
                {_is_alpha_boosted},
                out.index({_is_alpha_boosted}) * uniform
            );
        }

        return (_scale * out).view(shape);
    }

    torch::Tensor Gamma::entropy() const {
        throw std::runtime_error{"Not implemented."};
    }

    torch::Tensor Gamma::log_prob(const torch::Tensor &x) const {
        auto reshaped = x.view({-1});
        auto out = (_alpha - 1) * reshaped.log() - reshaped / _scale - _alpha * _scale.log() - torch::special::gammaln(_alpha);
        return out.view(x.sizes());
    }

    torch::Tensor Gamma::prob(const torch::Tensor &value) const {
        return log_prob(value).exp_();
    }

    void Gamma::include(std::shared_ptr<constraints::Base> constraint) {
        throw std::runtime_error{"Not implemented."};
    }

    std::unique_ptr<Base> Gamma::index(const std::vector<torch::indexing::TensorIndex> &indexing) const {
        return std::make_unique<Gamma>(_alpha.index(indexing), _scale.index(indexing));
    }
}
