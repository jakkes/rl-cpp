#include "rl/utils/reward/backpropagate.h"


using namespace torch::indexing;

namespace rl::utils::reward
{
    torch::Tensor backpropagate(const torch::Tensor &rewards, float discount)
    {
        auto historysize = rewards.size(-1);
        
        auto out = torch::zeros_like(rewards);
        out.index_put_({"...", -1}, rewards.index({"...", -1}));

        for (int64_t i = historysize - 2; i >= 0; i--)
        {
            out.index_put_(
                {"...", i},
                rewards.index({"...", i}) + discount * out.index({"...", i+1})
            );
        }

        return out;
    }
}
