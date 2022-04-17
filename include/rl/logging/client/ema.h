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
    /**
     * @brief Simple logging client computing the exponential moving average (EMA)
     * across different smoothing values, and outputting the smoothed values to STDOUT.
     * 
     */
    class EMA : public Base
    {
        public:
            /**
             * @brief Construct a new EMA object
             * 
             * @param smoothing_values Smoothing values to be used in the logging.
             * @param output_period_s Period, in seconds, of the output.
             */
            EMA(const std::vector<double> &smoothing_values, int output_period_s);
            ~EMA();
            
            /// <inheritdoc/>
            void log_scalar(const std::string &name, double value) override;

            /// <inheritdoc/>
            void log_frequency(const std::string &name, int occurances) override;

            /**
             * @brief EMA estimator.
             * 
             */
            class Estimator {
                public:
                    /**
                     * @brief Construct a new Estimator object
                     * 
                     * @param smoothing EMA smoothing value.
                     */
                    Estimator(double smoothing)
                    : smoothing{smoothing}, _x{smoothing}, inv_smoothing{1 - smoothing}
                    {}

                    /**
                     * @return double Current estimate.
                     */
                    inline double get() const { return value / (1 - _x); }
                    
                    /**
                     * @brief Update estimate.
                     */
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
            rl::cpputils::Metronome<std::chrono::seconds> metronome{5};

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
