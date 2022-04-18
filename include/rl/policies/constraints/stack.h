#ifndef INCLUDE_RL_POLICIES_CONSTRAINTS_STACK_H_
#define INCLUDE_RL_POLICIES_CONSTRAINTS_STACK_H_

#include <memory>
#include <functional>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "base.h"


namespace rl::policies::constraints
{

    template<class T>
    std::unique_ptr<T> __stack_impl(const std::vector<std::shared_ptr<T>> &constraints)
    {
        throw std::runtime_error{"Stack fn not implemented."};
    }

    template<class T>
    std::unique_ptr<Base> stack(const std::vector<std::shared_ptr<Base>> &constraints)
    {
        std::vector<std::shared_ptr<T>> recast{};
        recast.reserve(constraints.size());
        for (auto ptr : constraints) {
            auto casted_ptr = std::dynamic_pointer_cast<T>(ptr); assert(casted_ptr);
            recast.push_back(casted_ptr);
        }
        return __stack_impl<T>(recast);
    }

    std::unique_ptr<Base> stack(const std::vector<std::shared_ptr<Base>> &constraints);
}


#endif /* INCLUDE_RL_POLICIES_CONSTRAINTS_STACK_H_ */
