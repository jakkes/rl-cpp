#include "rl/agents/dqn/modules/distributional.h"

#include <exception>

namespace rl::agents::dqn::modules
{
    DistributionalOutput::DistributionalOutput(
        const torch::Tensor &logits,
        const torch::Tensor &atoms,
        float v_min,
        float v_max
    )
    : 
    logits{logits},
    atoms{atoms},
    v_min{v_min},
    v_max{v_max}
    {
        n_atoms = atoms.size(0);
        dz = (v_max - v_min) / (n_atoms - 1);
    }

    const torch::Tensor DistributionalOutput::value() const {
        auto out = (atoms * torch::softmax(logits, -1)).sum(-1);
        if (mask_set) {
            out = out.index_put({inverted_mask}, torch::zeros({inverted_mask.sum().item().toLong()}, out.options()) - INFINITY);
        }
        return out;
    }

    void DistributionalOutput::apply_mask(const torch::Tensor &mask) {
        inverted_mask = ~mask;
        mask_set = true;
    }

    torch::Tensor DistributionalOutput::loss(
        const torch::Tensor &actions,
        const torch::Tensor &rewards,
        const torch::Tensor &not_terminals,
        const BaseOutput &next_output,
        const torch::Tensor &next_actions,
        float discount
    )
    {
        const auto &next_output_ = dynamic_cast<const DistributionalOutput&>(next_output);
        int64_t batch_size = actions.size(0);
        torch::Tensor batchvec = torch::arange(batch_size, torch::TensorOptions{}.device(actions.device()));

        // Check validity of next actions -- either mask is not set or all next actions are of value "false" in the inverted mask
        assert (!next_output_.mask_set || !next_output_.inverted_mask.index({batchvec, next_actions}).any().item().toBool());

        auto next_distributions = torch::softmax(next_output_.logits.index({batchvec, next_actions}), -1);

        auto m = torch::zeros({batch_size, n_atoms}, rewards.options());

        auto projection = rewards.view({-1, 1}) + not_terminals.view({-1, 1}) * discount * atoms.view({1, -1});
        projection.clamp_(v_min, v_max);

        auto b = (projection - v_min) / dz;

        auto lower = b.floor().to(torch::kLong).clamp_(0, n_atoms - 1);
        auto upper = b.ceil().to(torch::kLong).clamp_(0, n_atoms - 1);

        auto lower_eq_upper = lower == upper;
        if (lower_eq_upper.any().item().toBool()) {
            auto lower_mask = (upper > 0).logical_and_(lower_eq_upper);
            lower.index_put_({lower_mask}, lower.index({lower_mask}) - 1);
        }

        lower_eq_upper = lower == upper;
        if (lower_eq_upper.any().item().toBool()) {
            auto upper_mask = (lower < n_atoms - 1).logical_and_(lower_eq_upper);
            upper.index_put_({upper_mask}, upper.index({upper_mask}) + 1);
        }

        auto index_vec_0 = batchvec.view({-1, 1}).repeat({1, n_atoms}).view({-1});
        auto index_vec_1a = lower.view({-1});

        m.index_put_(
            {
                index_vec_0,
                lower.view({-1})
            },
            (next_distributions * (upper - b)).view({-1}),
            true
        );

        m.index_put_(
            {
                index_vec_0,
                upper.view({-1})
            },
            (next_distributions * (b - lower)).view({-1}),
            true
        );
        auto current_logits = logits.index({batchvec, actions});
        auto log_distribution = torch::log_softmax(current_logits, -1);
        return - (m * log_distribution).sum(-1);
    }
}
