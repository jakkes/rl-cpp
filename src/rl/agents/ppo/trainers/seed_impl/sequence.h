#ifndef RL_AGENTS_PPO_TRAINERS_SEED_IMPL_SEQUENCE_H_
#define RL_AGENTS_PPO_TRAINERS_SEED_IMPL_SEQUENCE_H_

#include <vector>


#include <torch/torch.h>

#include "rl/policies/constraints/base.h"

namespace rl::agents::ppo::trainers::seed_impl
{
    struct Sequence {
        std::vector<torch::Tensor> states{};
        std::vector<torch::Tensor> actions{};
        std::vector<float> rewards{};
        std::vector<uint8_t> not_terminals{};
        std::vector<torch::Tensor> action_probabilities{};
        std::vector<torch::Tensor> state_values{};
        std::vector<std::shared_ptr<policies::constraints::Base>> constraints{};

        Sequence() {}

        Sequence(int length)
        {
            states.reserve(length + 1);
            actions.reserve(length);
            rewards.reserve(length);
            not_terminals.reserve(length);
            action_probabilities.reserve(length);
            state_values.reserve(length);   // Final state values not necessary
            constraints.reserve(length + 1);
        }
    };
}

#endif /* RL_AGENTS_PPO_TRAINERS_SEED_IMPL_SEQUENCE_H_ */
