#include "rl/policies/constraints/cat.h"

#include "rl/cpputils.h"


namespace rl::policies::constraints
{
    Concat::Concat(const std::vector<std::shared_ptr<Base>> &constraints)
    : constraints{constraints}
    {
        if (constraints.empty()) throw std::invalid_argument{"Empty constraint list."};
    }

    Concat::Concat(std::initializer_list<std::shared_ptr<Base>> constraints)
    :  Concat{std::vector<std::shared_ptr<Base>>(constraints.begin(), constraints.end())}
    {}

    void Concat::push_back(std::shared_ptr<Base> constraint)
    {
        constraints.push_back(constraint);
    }

    torch::Tensor Concat::contains(const torch::Tensor &value) const
    {
        auto re = constraints[0]->contains(value);
        
        for (int i = 1; i < constraints.size(); i++) {
            re.logical_and_(constraints[i]->contains(value));
        }

        return re;
    }
}
