#include "rl/agents/ppo/trainers/basic.h"


using namespace torch::indexing;

namespace rl::agents::ppo::trainers
{
    static
    torch::Tensor policy_loss(torch::Tensor A, torch::Tensor old_probs, torch::Tensor new_probs, float eps)
    {
        auto pr = new_probs / old_probs;
        auto clipped = pr.clamp(1-eps, 1+eps);
        pr = pr * A;
        clipped = clipped * A;
        return - torch::min(clipped, pr).mean();
    }

    static
    torch::Tensor compute_deltas(torch::Tensor rewards, torch::Tensor V, torch::Tensor not_terminals, float discount)
    {
        return rewards + discount * not_terminals * V.index({"...", Slice(1, None)}).detach() - V.index({"...", Slice(None, -1)});
    }

    static
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

    Basic::Basic(
        rl::agents::ppo::Module model,
        std::unique_ptr<torch::optim::Optimizer> optimizer,
        std::unique_ptr<rl::env::Base> env,
        const BasicOptions &options
    ) : model{model}, optimizer{std::move(optimizer)},
        env{std::move(env)},options{options}
    {}

    template<class Rep, class Period>
    void Basic::run(std::chrono::duration<Rep, Period> duration)
    {
        auto start = std::chrono::steady_clock::now();
        auto end = start + duration;

        while (std::chrono::steady_clock::now() < end)
        {
            
        }
    }
}
