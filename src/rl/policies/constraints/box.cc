#include "rl/policies/constraints/box.h"


namespace rl::policies::constraints
{
    Box::Box(torch::Tensor lower_bound, torch::Tensor upper_bound, const BoxOptions &options)
    : lower{lower_bound}, upper{upper_bound}, options{options}
    {}

    torch::Tensor Box::contains(const torch::Tensor &x) const {
        auto upper_fulfilled = options.inclusive_upper ? x <= upper : x < upper;
        auto lower_fulfilled = options.inclusive_lower ? x <= lower : x < lower;

        return upper_fulfilled.logical_and_(lower_fulfilled);
    }
}
