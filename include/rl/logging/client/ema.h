#ifndef RL_LOGGING_CLIENT_EMA_H_
#define RL_LOGGING_CLIENT_EMA_H_


#include <vector>
#include <thread>
#include <map>
#include <atomic>
#include <mutex>

#include <thread_safe/collections/collections.h>

#include "base.h"

namespace rl::logging::client
{
    class EMA : public Base
    {
        public:
            EMA(const std::vector<float> &smoothing_values, int output_period_s);
            ~EMA();
            
            void log_scalar(const std::string &name, float value) override;

            class Estimator {
                public:
                    Estimator(float smoothing) : smoothing{smoothing}, _x{smoothing}, inv_smoothing{1 - smoothing} {}

                    inline float get() const { return value / (1 - _x); }
                    inline void update(float observation) {
                        value += inv_smoothing * (observation - value);
                        _x *= smoothing;
                    }

                private:
                    const float inv_smoothing;
                    const float smoothing;
                    float value{0.0};
                    float _x;
            };

        private:
            std::mutex estimate_update_mtx{};
            std::map<std::string, std::vector<Estimator>> estimates;
            
            std::vector<float> smoothing_values;
            int output_period;
            
            thread_safe::Queue<std::pair<std::string, float>> scalar_queue{1000};
            std::thread queue_consuming_thread;
            std::thread output_producing_thread;

            std::atomic<bool> is_running{true};
            void queue_consumer();
            void output_producer();
            void new_log_name(const std::string &name);
    };
}

#endif /* RL_LOGGING_CLIENT_EMA_H_ */
