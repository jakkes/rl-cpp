#ifndef RL_AGENTS_ALPHA_ZERO_MCTS_H_
#define RL_AGENTS_ALPHA_ZERO_MCTS_H_


#include <torch/torch.h>

namespace rl::agents::alpha_zero
{
    struct MCTSNode
    {
        MCTSNode(const torch::Tensor &priors);

        torch::Tensor Q;
        torch::Tensor N;
        torch::Tensor P;
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_MCTS_H_ */
