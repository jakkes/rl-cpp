#ifndef RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_SELF_PLAY_EPISODE_H_
#define RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_SELF_PLAY_EPISODE_H_


#include <torch/torch.h>

namespace trainer_impl
{
    struct SelfPlayEpisode
    {
        torch::Tensor states;
        torch::Tensor masks;
        torch::Tensor collected_rewards;
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_SELF_PLAY_EPISODE_H_ */
