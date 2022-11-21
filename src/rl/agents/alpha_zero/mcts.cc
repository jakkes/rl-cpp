#include "rl/agents/alpha_zero/mcts.h"


namespace rl::agents::alpha_zero
{
    MCTSNode::MCTSNode(const torch::Tensor &priors)
        : 
        P{priors},
        Q{torch::zeros_like(priors)},
        N{torch::zeros_like(priors).to(torch::kLong)}
    {}
}
