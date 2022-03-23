#ifndef RL_LOGGING_CLIENT_EMA_H_
#define RL_LOGGING_CLIENT_EMA_H_


#include <vector>
#include <thread>
#include <unordered_map>
#include <atomic>

#include <thread_safe/collections/collections.h>

#include "base.h"

namespace rl::logging::client
{
    class EMA : public Base
    {
        public:
            EMA(const std::vector<float> &values, int update_period_s);
            ~EMA();
            
            void log_scalar(const std::string &name, float value) override;

        private:
            std::unordered_map<std::string, std::vector<float>> estimates;
            std::vector<float> values;
            int update_period;
            
            thread_safe::Queue<std::pair<std::string, float>> scalar_queue{1000};
            std::thread queue_consuming_thread;
            std::thread output_producing_thread;

            std::atomic<bool> is_running{true};
            void queue_consumer();
            void output_producer();
    };
}

#endif /* RL_LOGGING_CLIENT_EMA_H_ */
