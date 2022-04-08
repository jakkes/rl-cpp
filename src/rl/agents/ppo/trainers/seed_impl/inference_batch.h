#ifndef RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_BATCH_H_
#define RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_BATCH_H_


#include <mutex>
#include <exception>
#include <atomic>

#include "rl/env/state.h"
#include "rl/policies/policies.h"
#include "rl/agents/ppo/module.h"

#include "inference_options.h"

namespace rl::agents::ppo::trainers::seed_impl
{

    class InferenceBatchObsolete : public std::exception {};

    class InferenceBatch
    {
        public:
            InferenceBatch(
                std::shared_ptr<rl::agents::ppo::Module> model,
                InferenceOptions *options
            );

            inline bool is_full() { return size >= options->batchsize; }
            inline bool has_executed() { return executed; }
            inline bool is_empty() { return size == 0; }

            int64_t add_inference_request(const rl::env::State &state);
            torch::Tensor get_inference_result(int64_t label);

        private:
            const std::shared_ptr<rl::agents::ppo::Module> model;
            const InferenceOptions *options;

            bool executed{false};
            std::atomic<int64_t> size{0};
            std::vector<torch::Tensor> states;
            std::vector<std::shared_ptr<rl::policies::constraints::Base>> constraints;
            torch::Tensor result;

            std::mutex add_mtx{};

            void execute();
            void start_timer();
    };
}
#endif /* RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_BATCH_H_ */
