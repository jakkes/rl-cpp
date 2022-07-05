#ifndef RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_OPTIONS_H_
#define RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_OPTIONS_H_


#include <memory>

#include <torch/torch.h>

#include "rl/option.h"
#include "rl/logging/client/base.h"

namespace rl::agents::ppo::trainers::seed_impl
{
    struct InferenceOptions
    {
        RL_OPTION(int, batchsize) = 32;
        RL_OPTION(int, max_delay_ms) = 500;
        RL_OPTION(torch::Device, device) = torch::kCPU;

        RL_OPTION(std::shared_ptr<rl::logging::client::Base>, logger) = nullptr;
    };
}


#endif /* RL_AGENTS_PPO_TRAINERS_SEED_IMPL_INFERENCE_OPTIONS_H_ */
