#ifndef RL_AGENTS_UTILS_SEED_INFERENCE_H_
#define RL_AGENTS_UTILS_SEED_INFERENCE_H_

#include <mutex>
#include <memory>
#include <vector>

#include <torch/torch.h>
#include <thread_safe/collections/queue.h>

#include "rl/option.h"
#include "rl/policies/policies.h"
#include "rl/env/env.h"


namespace rl::agents::utils::seed
{

    class __InferenceBatch
    {
        public:
            __InferenceBatch(std::shared_ptr<torch::nn::Module> model, int capacity, int max_delay_ms);

            torch::Tensor get_action(const rl::env::State &state);

            inline bool is_full() { return states.size() >= capacity; }

        private:
            const int capacity;
            const int max_delay_ms;
            std::shared_ptr<torch::nn::Module> model;

            bool has_executed{false};
            std::vector<torch::Tensor> states;
            std::vector<std::shared_ptr<rl::policies::constraints::Base>> constraints;
            torch::Tensor result;

            void execute();
    };

    struct InferenceWorkerOptions
    {
        // Max batchsize of inference executions
        RL_OPTION(int, max_batchsize) = 32;
        
        // Max delay, in milliseconds, before a batch is executed.
        RL_OPTION(int, max_delay_ms) = 500;
    };

    class InferenceWorker
    {
        public:
            InferenceWorker(
                std::shared_ptr<torch::nn::Module> model,
                const InferenceWorkerOptions &options={}
            );

            torch::Tensor get_action(const rl::env::State &state);

        private:
            std::shared_ptr<torch::nn::Module> model;
            InferenceWorkerOptions options;

            std::mutex add_mtx{};
            std::shared_ptr<__InferenceBatch> current_batch{};

            int add_to_batch(const rl::env::State &state);
    };
}

#endif /* RL_AGENTS_UTILS_SEED_INFERENCE_H_ */
