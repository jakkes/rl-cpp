#include "loss_fns.h"


using namespace torch::indexing;

namespace rl::agents::ppo::trainers
{
    torch::Tensor compute_policy_loss(torch::Tensor A, torch::Tensor old_probs, torch::Tensor new_probs, float eps)
    {
        auto pr = new_probs / old_probs;
        auto clipped = pr.clamp(1-eps, 1+eps);
        pr = pr * A;
        clipped = clipped * A;
        return - torch::min(clipped, pr).mean();
    }

    torch::Tensor compute_deltas(torch::Tensor rewards, torch::Tensor V, torch::Tensor not_terminals, float discount)
    {
        return rewards + discount * not_terminals * V.index({"...", Slice(1, None)}).detach() - V.index({"...", Slice(None, -1)});
    }

    torch::Tensor compute_advantages(torch::Tensor deltas, torch::Tensor not_terminals, float discount, float gae_discount)
    {
        float d = discount * gae_discount;
        auto A = torch::empty_like(deltas);

        A.index_put_({"...", -1}, deltas.index({"...", -1}));
        for (int k = A.size(1) - 2; k > -1; k--) {
            A.index_put_({"...", k}, deltas.index({"...", k}) + d * not_terminals.index({"...", k}) * A.index({"...", k + 1}));
        }

        return A;
    }

    torch::Tensor compute_value_loss(torch::Tensor deltas)
    {
        return deltas.square().mean();
    }
}
