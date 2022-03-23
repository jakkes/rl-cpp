#include "rl/logging/client/ema.h"


namespace rl::logging::client
{
    EMA::EMA(const std::vector<float> &values, int update_period_s)
    : values{values}, update_period{update_period_s}
    {
        queue_consuming_thread = std::thread(&EMA::queue_consumer, this);
    }

    EMA::~EMA()
    {
        is_running = false;
        if (queue_consuming_thread.joinable()) queue_consuming_thread.join();
    }

    void EMA::log_scalar(const std::string &name, float value)
    {
        scalar_queue.enqueue({name, value});
    }

    void EMA::queue_consumer()
    {
        while (is_running)
        {
            auto log = scalar_queue.dequeue(std::chrono::seconds(1));
            if (!log) continue;

            
        }
    }
}