#ifndef RL_SIMULATORS_BASE_H_
#define RL_SIMULATORS_BASE_H_


#include <memory>

#include "transition.h"

namespace rl::simulators
{
    class Base
    {
        public:
            virtual ~Base() = default;


    };

    class Factory
    {
        public:
            virtual std::unique_ptr<Base> get() const = 0;
    };
}

#endif /* RL_SIMULATORS_BASE_H_ */
