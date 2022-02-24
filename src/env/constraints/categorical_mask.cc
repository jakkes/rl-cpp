#include "rl/env/constraints/categorical_mask.h"

#include <stdexcept>


namespace rl::env::constraints
{
    CategoricalMask::CategoricalMask(torch::Tensor mask)
    : mask{mask} {}

    const torch::Tensor CategoricalMask::contains(const torch::Tensor value) const
    {
        
    }
}
