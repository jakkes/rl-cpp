#ifndef RL_UTILS_FLOAT_CONTROL_BASE_H_
#define RL_UTILS_FLOAT_CONTROL_BASE_H_


namespace rl::utils::float_control
{
    class Base
    {
        public:
            virtual float get() const = 0;
    };
}

#endif /* RL_UTILS_FLOAT_CONTROL_BASE_H_ */
