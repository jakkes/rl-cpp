#include "rl/policies/constraints/categorical_mask.h"

#include <stdexcept>
#include <set>

#include "rl/torchutils.h"


namespace rl::policies::constraints
{
    static std::set<torch::ScalarType> allowed_dtypes{{ torch::kInt64, torch::kInt32, torch::kInt16, torch::kInt8 }};

    CategoricalMask::CategoricalMask(const torch::Tensor &mask)
    : mask{mask} {}

    const torch::Tensor CategoricalMask::contains(const torch::Tensor &value) const
    {
        if (!rl::torchutils::is_int_dtype(value)) {
            throw std::runtime_error{"Categorical mask received value of unknown data type."};
        }
        return torch::tensor({0, 1});
    }
}
