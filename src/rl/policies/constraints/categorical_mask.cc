#include "rl/policies/constraints/categorical_mask.h"

#include <stdexcept>
#include <set>

#include <rl/torchutils/torchutils.h>
#include "rl/cpputils/concat_vector.h"


namespace rl::policies::constraints
{
    CategoricalMask::CategoricalMask(const torch::Tensor &mask)
    :
    _mask{register_buffer("mask", mask)},
    dim{mask.size(-1)}
    {
        if (!rl::torchutils::is_bool_dtype(mask)) {
            throw std::invalid_argument{"Mask must be of type boolean."};
        }
        if (mask.sizes().size() < 1) {
            throw std::invalid_argument{"Mask must have at least one dimension."};
        }
        if (mask.isnan().any().item().toBool()) {
            throw std::runtime_error{"Mask must not contain NaN."};
        }
    }

    torch::Tensor CategoricalMask::contains(const torch::Tensor &value) const
    {
        if (!rl::torchutils::is_int_dtype(value)) {
            throw std::runtime_error{"Categorical mask received value of unknown data type."};
        }

        auto shape = value.sizes();
        auto broadcasted_and_flattened_mask = _mask
            .broadcast_to(
                rl::cpputils::concat<int64_t>(
                    value.sizes().vec(),
                    {dim}
                )
            )
            .reshape({-1, dim});
        auto batchvec = torch::arange(
                            broadcasted_and_flattened_mask.size(0), value.options());

        return broadcasted_and_flattened_mask
            .index({batchvec, value.view({-1})})
            .view(shape);
    }

    std::unique_ptr<Base> CategoricalMask::index(
                    const std::vector<torch::indexing::TensorIndex> &indexing) const
    {
        return std::make_unique<CategoricalMask>(_mask.index(indexing));
    }

    std::unique_ptr<Base> CategoricalMask::stack(const std::vector<std::shared_ptr<Base>> &constraints) const
    {
        return stack( *recast<CategoricalMask>(constraints) );
    }

    std::unique_ptr<CategoricalMask> CategoricalMask::stack(
                    const std::vector<std::shared_ptr<CategoricalMask>> &constraints) const
    {
        std::vector<torch::Tensor> masks;
        masks.reserve(constraints.size());

        for (auto &constraint : constraints) {
            masks.push_back(constraint->mask());
        }

        return std::make_unique<CategoricalMask>(torch::stack(masks));
    }
}
