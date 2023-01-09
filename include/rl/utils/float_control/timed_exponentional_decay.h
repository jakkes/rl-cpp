#ifndef RL_UTILS_FLOAT_CONTROL_TIMED_EXPONENTIONAL_DECAY_H_
#define RL_UTILS_FLOAT_CONTROL_TIMED_EXPONENTIONAL_DECAY_H_


#include <chrono>
#include <cmath>

#include "base.h"


namespace rl::utils::float_control
{
    class TimedExponentialDecay : public Base
    {
        public:
            TimedExponentialDecay(float start, float end, float half_life_seconds)
            : start{start}, end{end}, half_life_seconds{half_life_seconds}
            {}

            inline
            float get() const override {
                auto delta_s = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time_point).count();
                return end + (start - end) * std::pow(0.5f, delta_s / half_life_seconds);
            }

        private:
            float start, end, half_life_seconds;
            std::chrono::_V2::system_clock::time_point start_time_point{std::chrono::high_resolution_clock::now()};
    };
}

#endif /* RL_UTILS_FLOAT_CONTROL_TIMED_EXPONENTIONAL_DECAY_H_ */
