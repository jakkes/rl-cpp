#include "rl/policies/constraints/stack.h"

#include <memory>

#include "rl/policies/constraints/constraints.h"


namespace rl::policies::constraints
{
    std::unique_ptr<Base> stack(const std::vector<std::shared_ptr<Base>> &constraints)
    {
        if (constraints.size() == 0) throw std::invalid_argument{"Cannot stack zero elements."};
        
        // If no constraints, return no constraint.
        if (!constraints[0]) return nullptr;

        return constraints[0]->stack(constraints);
    }
}