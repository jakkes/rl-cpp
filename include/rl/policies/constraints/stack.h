#ifndef INCLUDE_RL_POLICIES_CONSTRAINTS_STACK_H_
#define INCLUDE_RL_POLICIES_CONSTRAINTS_STACK_H_

#include <memory>
#include <functional>
#include <vector>
#include <stdexcept>

#include "base.h"


namespace rl::policies::constraints
{
    template<class T>
    std::unique_ptr<std::vector<std::shared_ptr<T>>> recast(
        const std::vector<std::shared_ptr<Base>> &constraints
    )
    {
        auto out = std::make_unique<std::vector<std::shared_ptr<T>>>();
        out->reserve(constraints.size());
        for (auto ptr : constraints) {
            auto casted_ptr = std::dynamic_pointer_cast<T>(ptr);
            if (!casted_ptr) throw std::runtime_error{"Failed casting constraint."};
            out->push_back(casted_ptr);
        }
        return out;
    }

    std::unique_ptr<Base> stack(const std::vector<std::shared_ptr<Base>> &constraints);
}


#endif /* INCLUDE_RL_POLICIES_CONSTRAINTS_STACK_H_ */
