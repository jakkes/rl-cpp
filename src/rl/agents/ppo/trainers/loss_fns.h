#ifndef RL_AGENTS_PPO_TRAINERS_LOSS_FNS_H_
#define RL_AGENTS_PPO_TRAINERS_LOSS_FNS_H_


#include <torch/torch.h>

namespace rl::agents::ppo::trainers
{
    torch::Tensor compute_policy_loss(torch::Tensor A, torch::Tensor old_probs, torch::Tensor new_probs, float eps);

    torch::Tensor compute_deltas(torch::Tensor rewards, torch::Tensor V, torch::Tensor not_terminals, float discount);

    torch::Tensor compute_advantages(torch::Tensor deltas, torch::Tensor not_terminals, float discount, float gae_discount);

    torch::Tensor compute_value_loss(torch::Tensor deltas);
}

#endif /* RL_AGENTS_PPO_TRAINERS_LOSS_FNS_H_ */
