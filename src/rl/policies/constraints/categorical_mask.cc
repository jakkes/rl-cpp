#include "rl/policies/constraints/categorical_mask.h"

#include <stdexcept>
#include <set>

#include "rl/torchutils.h"


namespace rl::policies::constraints
{
    CategoricalMask::CategoricalMask(const torch::Tensor &mask)
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

        batch = mask.sizes().size() > 1;
        dim = mask.size(-1);

        if (batch) {
            this->_mask = mask.view({-1, dim});
            batchsize = mask.size(0);
        } else {
            this->_mask = mask;
        }

        batchvec = torch::arange(batchsize, torch::TensorOptions{}.device(mask.device()));
    }

    torch::Tensor CategoricalMask::contains(const torch::Tensor &value) const
    {
        if (!rl::torchutils::is_int_dtype(value)) {
            throw std::runtime_error{"Categorical mask received value of unknown data type."};
        }
        if (batch) {
            assert(value.sizes().size() > 0);
            auto shape = value.sizes();
            return _mask.index({batchvec, value.view({-1})}).view(shape);
        }
        else
        {
            return _mask.index({value});
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
            masks.push_back(constraint->_mask);
        }

        return std::make_unique<CategoricalMask>(torch::stack(masks));
    }
}
