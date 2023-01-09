#ifndef RL_AGENTS_ALPHA_ZERO_SELF_PLAY_EPISODE_H_
#define RL_AGENTS_ALPHA_ZERO_SELF_PLAY_EPISODE_H_


#include <torch/torch.h>


namespace rl::agents::alpha_zero
{
    struct SelfPlayEpisode
    {
        torch::Tensor states;
        torch::Tensor masks;
        torch::Tensor collected_rewards;
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_SELF_PLAY_EPISODE_H_ */
