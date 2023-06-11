#ifndef RL_AGENTS_ALPHA_ZERO_SELF_PLAY_EPISODE_H_
#define RL_AGENTS_ALPHA_ZERO_SELF_PLAY_EPISODE_H_


#include <torch/torch.h>


namespace rl::agents::alpha_zero
{
    struct SelfPlayEpisode
    {
        // Sequence of states in an episode.
        torch::Tensor states;
        // Sequence of masks in an episode.
        torch::Tensor masks;
        // Sequence of actions taken.
        torch::Tensor actions;
        // Sequence of (future, discounted) rewards collected during an episode, G.
        torch::Tensor collected_rewards;
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_SELF_PLAY_EPISODE_H_ */
