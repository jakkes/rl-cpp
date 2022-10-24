#ifndef RL_AGENTS_DQN_UTILS_HINDSIGHT_REPLAY_H_
#define RL_AGENTS_DQN_UTILS_HINDSIGHT_REPLAY_H_


#include <functional>
#include <vector>

#include <torch/torch.h>

#include <rl/env/state.h>

namespace rl::agents::dqn::utils
{
    /**
     * @brief One episode of states, action masks, actions and rewards.
     * 
     * In an episode of N steps, this struct contains N+1 states, N+1 masks, N actions,
     * and N rewards. Terminal flags are implicitly defined as non terminal for steps
     * n = 1, 2, ..., N-1, and terminal for step N.
     */
    struct HindsightReplayEpisode
    {
        std::vector<std::shared_ptr<rl::env::State>> states;
        std::vector<torch::Tensor> actions;
        std::vector<float> rewards;
    };

    using HindsightReplayCallback = std::function<bool(HindsightReplayEpisode*)>;
}

#endif /* RL_AGENTS_DQN_UTILS_HINDSIGHT_REPLAY_H_ */
