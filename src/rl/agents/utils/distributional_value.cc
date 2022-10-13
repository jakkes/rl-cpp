#include "rl/agents/utils/distributional_value.h"


namespace rl::agents::utils
{
    torch::Tensor distributional_value_loss(
        const torch::Tensor &current_logits,
        const torch::Tensor &rewards,
        const torch::Tensor &not_terminals,
        const torch::Tensor &next_logits,
        const torch::Tensor &atoms,
        float discount
    )
    {
        auto batchsize = current_logits.size(0);
        auto batchvec = torch::arange(batchsize, torch::TensorOptions{}.device(current_logits.device()));
        auto n_atoms = atoms.size(0);
        auto dz = atoms.index({1}) - atoms.index({0});
        float v_min = atoms.index({0}).item().toFloat();
        float v_max = atoms.index({-1}).item().toFloat();

        auto m = torch::zeros({batchsize, n_atoms}, rewards.options());

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


        auto next_distribution = next_logits.softmax(-1);
        m.index_put_(
            {
                index_vec_0,
                lower.view({-1})
            },
            (next_distribution * (upper - b)).view({-1}),
            true
        );

        m.index_put_(
            {
                index_vec_0,
                upper.view({-1})
            },
            (next_distribution * (b - lower)).view({-1}),
            true
        );

        auto log_distribution = torch::log_softmax(current_logits, -1);
        return - (m * log_distribution).sum(-1);
    }
}
