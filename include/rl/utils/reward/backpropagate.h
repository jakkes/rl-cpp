#ifndef RL_UTILS_REWARD_BACKPROPAGATE_H_
#define RL_UTILS_REWARD_BACKPROPAGATE_H_


#include <torch/torch.h>

namespace rl::utils::reward
{
    /**
     * @brief Computes the reversed cumulative reward sum, with discount factor.
     * 
     * @param rewards History of rewards of shape (N, H), where N is the batch size and
     *  H the history length. Cumulative sum is computed along the last dimension.
     * @param discount discount factor
     * @return torch::Tensor Tensor of same shape as input, with future discounted
     *  reward.
     */
    torch::Tensor backpropagate(const torch::Tensor &rewards, float discount);
}

#endif /* RL_UTILS_REWARD_BACKPROPAGATE_H_ */
