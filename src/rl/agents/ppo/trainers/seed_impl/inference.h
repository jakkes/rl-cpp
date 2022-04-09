#ifndef RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_H_
#define RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_H_


#include <atomic>
#include <memory>

#include "rl/option.h"
#include "rl/agents/ppo/module.h"
#include "rl/env/env.h"

#include "inference_options.h"
#include "inference_batch.h"
#include "inference_result_future.h"


namespace rl::agents::ppo::trainers::seed_impl
{
    class Inference
    {
        public:
            Inference(
                std::shared_ptr<rl::agents::ppo::Module> model,
                const InferenceOptions &options
            );

            std::unique_ptr<InferenceResultFuture> infer(const rl::env::State &state);

        private:
            const std::shared_ptr<rl::agents::ppo::Module> model;
            const InferenceOptions options;

            std::shared_ptr<InferenceBatch> current_batch;
            std::mutex current_batch_mtx{};

            std::shared_ptr<InferenceBatch> get_batch();
    };
}

#endif /* RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_H_ */
