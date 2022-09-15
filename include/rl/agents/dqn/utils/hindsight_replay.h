#ifndef RL_AGENTS_DQN_UTILS_HINDSIGHT_REPLAY_H_
#define RL_AGENTS_DQN_UTILS_HINDSIGHT_REPLAY_H_


#include <functional>
#include <vector>

#include <torch/torch.h>

namespace rl::agents::dqn::utils
{
    struct HindsightReplayEpisode
    {
        std::vector<torch::Tensor> states;
        std::vector<torch::Tensor> masks;
        std::vector<torch::Tensor> actions;
        std::vector<float> rewards;
    };

    using HindsightReplayCallback = std::function<bool(HindsightReplayEpisode*)>;
}

#endif /* RL_AGENTS_DQN_UTILS_HINDSIGHT_REPLAY_H_ */
