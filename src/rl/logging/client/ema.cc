#include "rl/logging/client/ema.h"

#include <algorithm>
#include <iostream>

namespace rl::logging::client
{

    EMA::EMA(const std::vector<double> &smoothing_values, int output_period_s, int frequency_window_seconds)
    :
    smoothing_values{smoothing_values},
    output_period{output_period_s},
    metronome{static_cast<size_t>(frequency_window_seconds)}
    {
        queue_consuming_thread = std::thread(&EMA::queue_consumer, this);
        output_producing_thread = std::thread(&EMA::output_producer, this);
        
        metronome.start(std::bind(&EMA::metronome_callback, this));
    }

    EMA::~EMA()
    {
        is_running = false;
        if (queue_consuming_thread.joinable()) queue_consuming_thread.join();
        if (output_producing_thread.joinable()) output_producing_thread.join();
    }

    void EMA::log_scalar(const std::string &name, double value)
    {
        scalar_queue.enqueue({name, value});
    }

    void EMA::log_frequency(const std::string &name, int occurences)
    {
        std::lock_guard lock{occurances_reset_mtx};
        this->occurences[name] += occurences;
    }

    void EMA::log_text(const std::string &name, const std::string &value) {
        std::lock_guard lock{text_mtx};
        text_logs[name] = value;
    }

    void EMA::queue_consumer()
    {
        while (is_running)
        {
            auto log = scalar_queue.dequeue(std::chrono::seconds(1));
            if (!log) continue;

            std::lock_guard lock{scalar_estimate_update_mtx};
            
            if (scalar_estimates.find(log->first) == scalar_estimates.end()) new_scalar_log_name(log->first);
            for (auto &estimate : scalar_estimates[log->first]) estimate.update(log->second);
        }
    }

    void EMA::new_scalar_log_name(const std::string &name) {
        scalar_estimates[name] = std::vector<Estimator>{};
        auto &vec = scalar_estimates[name];
        vec.reserve(smoothing_values.size());
        for (auto smoothing_value : smoothing_values) {
            vec.push_back(Estimator{smoothing_value});
        }
    }

    void EMA::output_producer()
    {
        while (is_running)
        {
            std::this_thread::sleep_for(std::chrono::seconds(output_period));

            auto print_estimates = [] (const std::pair<std::string, std::vector<Estimator>> &data) {
                std::cout << data.first << " -- ";
                for (auto &value : data.second) std::cout << value.get() << " ";
                std::cout << "\n";
            };

            auto print_text_logs = [] (const std::pair<std::string, std::string> &data) {
                std::cout << data.first << " -- " << data.second << "\n";
            };

            std::lock_guard lock{scalar_estimate_update_mtx};

            std::cout << "\n";
            std::for_each(scalar_estimates.cbegin(), scalar_estimates.cend(), print_estimates);
            std::cout << "\n";
            std::for_each(text_logs.cbegin(), text_logs.cend(), print_text_logs);
            std::cout << "\n";
            std::cout << std::flush;
        }
    }

    void EMA::metronome_callback()
    {
        std::lock_guard lock{occurances_reset_mtx};

        for (auto &pair : occurences) {
            log_scalar(pair.first, static_cast<double>(pair.second) / output_period);
            pair.second = 0;
        }
    }
}