#ifndef RL_AGENTS_UTILS_DISTRIBUTIONAL_LOSS_H_
#define RL_AGENTS_UTILS_DISTRIBUTIONAL_LOSS_H_


#include <torch/torch.h>

namespace rl::agents::utils
{
    torch::Tensor distributional_loss(
        const torch::Tensor &current_logits,
        const torch::Tensor &rewards,
        const torch::Tensor &not_terminals,
        const torch::Tensor &next_logits,
        const torch::Tensor &atoms,
        float discount,
        float v_min,
        float v_max,
        bool allow_cuda_graph=false
    );
}

#endif /* RL_AGENTS_UTILS_DISTRIBUTIONAL_LOSS_H_ */
