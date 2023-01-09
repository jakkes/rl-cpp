#ifndef RL_UTILS_FLOAT_CONTROL_FIXED_H_
#define RL_UTILS_FLOAT_CONTROL_FIXED_H_


#include "base.h"


namespace rl::utils::float_control
{
    class Fixed : public Base
    {
        public:
            Fixed(float value) : value{value} {}

            inline
            float get() const override { return value; }

        private:
            float value;
    };
}

#endif /* RL_UTILS_FLOAT_CONTROL_FIXED_H_ */
