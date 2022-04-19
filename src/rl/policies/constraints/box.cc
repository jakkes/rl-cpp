#include "rl/policies/constraints/box.h"

#include <algorithm>


namespace rl::policies::constraints
{
    bool BoxOptions::equals(const BoxOptions &other) const
    {
        return (
            inclusive_lower == other.inclusive_lower
            && inclusive_upper == other.inclusive_upper
            && n_action_dims == other.n_action_dims
        );
    }

    Box::Box(torch::Tensor lower_bound, torch::Tensor upper_bound, const BoxOptions &options)
    : lower{lower_bound}, upper{upper_bound}, options{options}
    {
        if (!torch::all(lower < upper).item().toBool()) {
            throw std::invalid_argument{"Lower bound must be smaller than upper bound."};
        }
    }

    torch::Tensor Box::contains(const torch::Tensor &x) const {
        auto upper_fulfilled = options.inclusive_upper ? x <= upper : x < upper;
        auto lower_fulfilled = options.inclusive_lower ? x >= lower : x > lower;

        for (int i = 0; i < options.n_action_dims; i++) {
            upper_fulfilled = upper_fulfilled.all({-1});
            lower_fulfilled = lower_fulfilled.all({-1});
        }

        return upper_fulfilled.logical_and_(lower_fulfilled);
    }

    std::unique_ptr<Base> Box::index(const std::vector<torch::indexing::TensorIndex> &indexing) const
    {
        return std::make_unique<Box>(lower.index(indexing), upper.index(indexing), options);
    }

    const torch::Tensor Box::lower_bound() const { return lower; }
    const torch::Tensor Box::upper_bound() const { return upper; }
    int Box::n_action_dims() const { return options.n_action_dims; }

    std::unique_ptr<Base> Box::stack(const std::vector<std::shared_ptr<Base>> &constraints) const
    {
        return stack( *recast<Box>(constraints) );
    }

    std::unique_ptr<Box> Box::stack(const std::vector<std::shared_ptr<Box>> &constraints) const
    {
        for (int i = 1; i < constraints.size(); i++) {
            if (!constraints[i]->options.equals(constraints[0]->options)) {
                throw std::invalid_argument{"Constraints do not have equal options."};
            }
        }

        std::vector<torch::Tensor> lower{}; lower.reserve(constraints.size());
        std::vector<torch::Tensor> upper{}; upper.reserve(constraints.size());

        for (const auto &constraint : constraints) {
            lower.push_back(constraint->lower);
            upper.push_back(constraint->upper);
        }

        return std::make_unique<Box>(
            torch::stack(lower), torch::stack(upper)
        );
    }
}
