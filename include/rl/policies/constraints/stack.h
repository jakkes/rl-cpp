#ifndef INCLUDE_RL_POLICIES_CONSTRAINTS_STACK_H_
#define INCLUDE_RL_POLICIES_CONSTRAINTS_STACK_H_

#include <memory>
#include <functional>
#include <vector>

#include "base.h"


namespace rl::policies::constraints
{
    template<class T>
    std::unique_ptr<T> stack(const std::vector<std::shared_ptr<T>> &constraints) {
        throw std::runtime_error{"Not implemented."};
    }

    template<class T>
    std::unique_ptr<T> stack(std::initializer_list<std::shared_ptr<T>> constraints) {
        return stack<T>(std::vector<std::shared_ptr<T>>(constraints.begin(), constraints.end()));
    }

    template<class T>
    std::unique_ptr<Base> __stack_recast(const std::vector<std::shared_ptr<Base>> &constraints)
    {
        std::vector<std::shared_ptr<T>> recast{};
        recast.reserve(constraints.size());
        for (auto ptr : constraints) {
            recast.push_back(std::dynamic_pointer_cast<T>(ptr));
        }
        return stack<T>(recast);
    }

    template<>
    std::unique_ptr<Base> stack<Base>(const std::vector<std::shared_ptr<Base>> &constraints);
}


#endif /* INCLUDE_RL_POLICIES_CONSTRAINTS_STACK_H_ */
