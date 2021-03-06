#ifndef RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_RESULT_FUTURE_H_
#define RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_RESULT_FUTURE_H_


#include <memory>

#include <torch/torch.h>

#include "inference_batch.h"


namespace rl::agents::ppo::trainers::seed_impl
{

    class InferenceResultFuture
    {
        public:
            InferenceResultFuture(
                std::shared_ptr<InferenceBatch> batch,
                int64_t label
            );

            inline
            bool is_ready() { return batch->has_executed(); }

            inline
            std::unique_ptr<InferenceResult> get() { return batch->get_inference_result(label); }

        private:
            const std::shared_ptr<InferenceBatch> batch;
            const int64_t label;
    };
}


#endif /* RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_RESULT_FUTURE_H_ */
