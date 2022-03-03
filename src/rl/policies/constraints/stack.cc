#include "rl/policies/constraints/stack.h"


namespace rl::policies::constraints
{
    template<>
    std::unique_ptr<Base> stack<Base>(const std::vector<std::shared_ptr<Base>> &constraints)
    {
        if (constraints.size() == 0) {
            throw std::runtime_error{"Cannot stack zero elements."};
        }
        return constraints[0]->stack_fn()(constraints);
    }
}
