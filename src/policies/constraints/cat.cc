#include "rl/policies/constraints/cat.h"


namespace rl::policies::constraints
{
    Concat::Concat(const std::vector<std::shared_ptr<Base>> &constraints)
    : constraints{constraints}
    {}

    void Concat::push_back(std::shared_ptr<Base> constraint)
    {
        constraints.push_back(constraint);
    }

    torch::Tensor Concat::contains(const torch::Tensor &value) const
    {
        return torch::randn({1,2});
    }
}
