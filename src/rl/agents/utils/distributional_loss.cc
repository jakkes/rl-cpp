#include "rl/agents/utils/distributional_loss.h"


using namespace torch::indexing;

namespace rl::agents::utils
{

    inline torch::Tensor get_m_cuda_graph_accum(
        int batchsize,
        int n_atoms,
        const torch::Tensor &next_distribution,
        const torch::Tensor &lower,
        const torch::Tensor &upper,
        const torch::Tensor &b
    )
    {
        auto m = torch::zeros_like(next_distribution);

        auto lower_add = next_distribution * (upper - b);
        auto upper_add = next_distribution * (b - lower);

        for (int i = 0; i < batchsize; i++) {
            for (int j = 0; j < n_atoms; j++) {
                auto k = lower.index({i, Slice(j, j+1)});
                auto l = upper.index({i, Slice(j, j+1)});

                m.index_put_(
                    {i, k},
                    m.index({i, k}) + lower_add.index({i, Slice(j, j+1)})
                );

                m.index_put_(
                    {i, l},
                    m.index({i, l}) + upper_add.index({i, Slice(j, j+1)})
                );
            }
        }
        return m;
    }

    inline torch::Tensor get_m_torch_vectorization(
        int batchsize,
        int n_atoms,
        const torch::Tensor &next_distribution,
        const torch::Tensor &lower,
        const torch::Tensor &upper,
        const torch::Tensor &b
    )
    {
        auto m = torch::zeros_like(next_distribution);
        auto batchvec = torch::arange(batchsize, torch::TensorOptions{}.device(next_distribution.device()));
        auto index_vec_0 = batchvec.view({-1, 1}).repeat({1, n_atoms}).view({-1});
        auto index_vec_1a = lower.view({-1});

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

        return m;
    }

    torch::Tensor distributional_loss(
        const torch::Tensor &current_logits,
        const torch::Tensor &rewards,
        const torch::Tensor &not_terminals,
        const torch::Tensor &next_logits,
        const torch::Tensor &atoms,
        float discount,
        bool allow_cuda_graph
    )
    {
        auto batchsize = current_logits.size(0);
        auto batchvec = torch::arange(batchsize, torch::TensorOptions{}.device(current_logits.device()));
        auto n_atoms = atoms.size(0);
        auto dz = atoms.index({1}) - atoms.index({0});
        auto v_min = atoms.index({0});
        auto v_max = atoms.index({-1});

        auto projection = rewards.view({-1, 1}) + not_terminals.view({-1, 1}) * discount * atoms.view({1, -1});
        projection.clamp_(v_min, v_max);
        auto b = (projection - v_min) / dz;

        auto lower = b.floor().to(torch::kLong).clamp_(0, n_atoms - 1);
        auto upper = b.ceil().to(torch::kLong).clamp_(0, n_atoms - 1);

        auto lower_eq_upper = lower == upper;
        auto lower_mask = (upper > 0).logical_and_(lower_eq_upper);
        lower = torch::where(lower_mask, lower - 1, lower);

        lower_eq_upper = lower == upper;
        auto upper_mask = (lower < n_atoms - 1).logical_and_(lower_eq_upper);
        upper = torch::where(upper_mask, upper + 1, upper);

        auto next_distribution = next_logits.softmax(-1);
        
        auto get_m_fn = allow_cuda_graph ? get_m_cuda_graph_accum : get_m_torch_vectorization;
        auto m = get_m_fn(batchsize, n_atoms, next_distribution, lower, upper, b);

        auto log_distribution = torch::log_softmax(current_logits, -1);
        return - (m * log_distribution).sum(-1);
    }
}
