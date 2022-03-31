#ifndef RL_LOGGING_CLIENT_EMA_H_
#define RL_LOGGING_CLIENT_EMA_H_


#include <vector>
#include <thread>
#include <unordered_map>
#include <atomic>
#include <mutex>

#include <thread_safe/collections/collections.h>

#include "rl/cpputils/metronome.h"
#include "base.h"


namespace rl::logging::client
{
    class EMA : public Base
    {
        public:
            EMA(const std::vector<double> &smoothing_values, int output_period_s);
            ~EMA();
            
            void log_scalar(const std::string &name, double value) override;
            void log_frequency(const std::string &name, int occurances) override;

            class Estimator {
                public:
                    Estimator(double smoothing)
                    : smoothing{smoothing}, _x{smoothing}, inv_smoothing{1 - smoothing}
                    {}

                    inline double get() const { return value / (1 - _x); }
                    inline void update(double observation) {
                        value += inv_smoothing * (observation - value);
                        _x *= smoothing;
                    }

                private:
                    const double inv_smoothing;
                    const double smoothing;
                    double value{0.0};
                    double _x;
            };

        private:
            std::mutex scalar_estimate_update_mtx{};
            std::unordered_map<std::string, std::vector<Estimator>> scalar_estimates;
            
            std::mutex occurances_reset_mtx{};
            std::unordered_map<std::string, std::atomic<size_t>> occurences;
            rl::cpputils::Metronome *metronome;

            std::vector<double> smoothing_values;
            int output_period;
            
            thread_safe::Queue<std::pair<std::string, double>> scalar_queue{1000};
            std::thread queue_consuming_thread;
            std::thread output_producing_thread;

            std::atomic<bool> is_running{true};
            void queue_consumer();
            void output_producer();
            void new_scalar_log_name(const std::string &name);

            void metronome_callback();
    };
}

#endif /* RL_LOGGING_CLIENT_EMA_H_ */
