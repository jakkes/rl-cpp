#include "rl/policies/constraints/cat.h"

#include "rl/cpputils.h"


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
        auto re = torch::ones(
            rl::cpputils::slice(value.sizes(), 0, -1),
            torch::TensorOptions{}
                .dtype(torch::kBool)
                .device(value.device())
        );

        for (const auto &constraint : constraints) {
            re.logical_and_(constraint->contains(value));
        }

        return re;
    }
}
