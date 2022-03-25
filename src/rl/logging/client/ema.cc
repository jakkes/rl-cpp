#include "rl/logging/client/ema.h"

#include <algorithm>
#include <iostream>

namespace rl::logging::client
{

    EMA::EMA(const std::vector<double> &smoothing_values, int output_period_s)
    : smoothing_values{smoothing_values}, output_period{output_period_s}
    {
        queue_consuming_thread = std::thread(&EMA::queue_consumer, this);
        output_producing_thread = std::thread(&EMA::output_producer, this);
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

    void EMA::queue_consumer()
    {
        while (is_running)
        {
            auto log = scalar_queue.dequeue(std::chrono::seconds(1));
            if (!log) continue;

            std::lock_guard lock{estimate_update_mtx};
            
            if (estimates.find(log->first) == estimates.end()) new_log_name(log->first);
            for (auto &estimate : estimates[log->first]) estimate.update(log->second);
        }
    }

    void EMA::new_log_name(const std::string &name) {
        estimates[name] = std::vector<Estimator>{};
        auto &vec = estimates[name];
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

            std::lock_guard lock{estimate_update_mtx};

            std::cout << "\n";
            std::for_each(estimates.cbegin(), estimates.cend(), print_estimates);
            std::cout << "\n";
        }
    }
}