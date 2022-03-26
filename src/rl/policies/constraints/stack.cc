#include "rl/policies/constraints/stack.h"

#include <memory>

#include "rl/policies/constraints/constraints.h"


namespace rl::policies::constraints
{
    std::unique_ptr<Base> stack(const std::vector<std::shared_ptr<Base>> &constraints)
    {
        if (constraints.size() == 0) throw std::invalid_argument{"Cannot stack zero elements."};

        if (std::dynamic_pointer_cast<CategoricalMask>(constraints[0])) {
            return stack<CategoricalMask>(constraints);
        }

        if (std::dynamic_pointer_cast<Box>(constraints[0])) {
            return stack<Box>(constraints);
        }

        throw UnknownConstraint{};
    }
}