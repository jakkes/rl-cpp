#ifndef RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_H_
#define RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_H_


#include <mutex>
#include <memory>

#include "rl/option.h"
#include "rl/agents/ppo/module.h"
#include "rl/env/env.h"

#include "inference_options.h"


namespace rl::agents::ppo::trainers::seed_impl
{
    class Inference
    {
        public:
            Inference(
                std::shared_ptr<rl::agents::ppo::Module> model,
                const InferenceOptions &options
            );

            torch::Tensor get_action(const rl::env::State &state);

        private:
            const std::shared_ptr<rl::agents::ppo::Module> model;
            const InferenceOptions options;
    };
}

#endif /* RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_H_ */
