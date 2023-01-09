#include "rl/utils/reward/backpropagate.h"


using namespace torch::indexing;

namespace rl::utils::reward
{
    torch::Tensor backpropagate(const torch::Tensor &rewards, float discount)
    {
        auto batchsize = rewards.size(0);
        auto historysize = rewards.size(1);
        
        auto out = torch::zeros_like(rewards);
        out.index_put_({Slice(), -1}, rewards.index({Slice(), -1}));

        for (int64_t i = historysize - 2; i >= 0; i--)
        {
            out.index_put_(
                {Slice(), i},
                rewards.index({Slice(), i}) + discount * out.index({Slice(), i+1})
            );
        }

        return out;
    }
}
