#include "rl/policies/constraints/categorical_mask.h"

#include <stdexcept>
#include <set>

#include "rl/torchutils.h"


namespace rl::policies::constraints
{
    CategoricalMask::CategoricalMask(const torch::Tensor &mask)
    : mask{mask}, batch{mask.sizes().size() == 2}, batchsize{mask.size(0)}
    {
        if (!rl::torchutils::is_bool_dtype(mask)) {
            throw std::invalid_argument{"Mask must be of type boolean."};
        }
        if (mask.sizes().size() < 1) {
            throw std::invalid_argument{"Mask must have at least one dimension."};
        }
        if (mask.sizes().size() > 2) {
            throw std::invalid_argument{"Mask must have at most two dimensions."};
        }
        if (mask.isnan().any().item().toBool()) {
            throw std::runtime_error{"Mask must not contain NaN."};
        }

        batchvec = torch::arange(batchsize, torch::TensorOptions{}.device(mask.device()));
    }

    torch::Tensor CategoricalMask::contains(const torch::Tensor &value) const
    {
        if (!rl::torchutils::is_int_dtype(value)) {
            throw std::runtime_error{"Categorical mask received value of unknown data type."};
        }
        if (batch) {
            assert(value.sizes().size() == 1);
            return mask.index({batchvec, value});
        }
        else
        {
            return mask.index({value});
        }
    }

    std::function<std::unique_ptr<Base>(const std::vector<std::shared_ptr<Base>>&)> CategoricalMask::stack_fn() const
    {
        return __stack_recast<CategoricalMask>;
    }

    template<>
    std::unique_ptr<CategoricalMask> stack<CategoricalMask>(const std::vector<std::shared_ptr<CategoricalMask>> &constraints)
    {
        std::vector<torch::Tensor> masks;
        masks.reserve(constraints.size());

        for (auto &constraint : constraints) {
            masks.push_back(constraint->mask);
        }

        return std::make_unique<CategoricalMask>(torch::stack(masks));
    }
}
