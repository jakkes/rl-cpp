#include "rl/policies/constraints/empty.h"

#include "rl/cpputils/slice_vector.h"


namespace rl::policies::constraints
{
    Empty::Empty(int n_action_dims)
    : n_action_dims{n_action_dims} {}

    std::unique_ptr<Base> Empty::stack(const std::vector<std::shared_ptr<Base>> &constraints) const
    {
        return std::make_unique<Empty>(n_action_dims);
    }

    std::unique_ptr<Base> Empty::index(const std::vector<torch::indexing::TensorIndex> &constraints) const
    {
        return std::make_unique<Empty>(n_action_dims);
    }

    torch::Tensor Empty::contains(const torch::Tensor &x) const
    {
        std::vector<int64_t> shape;
        if (n_action_dims == 0) {
            shape = x.sizes().vec();
        }
        else {
            shape = rl::cpputils::slice(x.sizes().vec(), 0, -n_action_dims);
        }

        return torch::ones(
            shape,
            torch::TensorOptions{}
                .device(x.device()).dtype(torch::kBool)
        );
    }
}
