#ifndef RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_OPTIONS_H_
#define RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_OPTIONS_H_


#include "rl/option.h"

namespace rl::agents::ppo::trainers::seed_impl
{
    struct InferenceOptions
    {
        RL_OPTION(int, batchsize) = 32;
        RL_OPTION(int, max_delay_ms) = 500;
    };
}


#endif /* RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_OPTIONS_H_ */
