#ifndef RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_GET_MASK_H_
#define RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_GET_MASK_H_


#include <vector>
#include <memory>

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

#include <rl/policies/constraints/categorical_mask.h>
#include <rl/agents/alpha_zero/mcts.h>


namespace trainer_impl
{
    inline
    const torch::Tensor get_mask(const rl::policies::constraints::Base &constraint)
    {
        return constraint.as_type<rl::policies::constraints::CategoricalMask>().mask();
    }

    inline
    rl::policies::Categorical mcts_nodes_to_policy(
        const std::vector<std::shared_ptr<rl::agents::alpha_zero::MCTSNode>> &nodes,
        float temperature
    )
    {
        std::vector<torch::Tensor> visit_counts_vector{}; visit_counts_vector.reserve(nodes.size());
        for (const auto &node : nodes) {
            visit_counts_vector.push_back(node->visit_count());
        }
        auto visit_counts = torch::stack({visit_counts_vector}).to(torch::kFloat32);
        visit_counts.pow_(1.0f / temperature);

        return rl::policies::Categorical{visit_counts};
    }

    inline
    std::vector<c10::Stream> get_cuda_streams()
    {
        std::vector<c10::Stream> out{};
        for (int i = 0; i < c10::cuda::device_count(); i++) {
            out.push_back(c10::cuda::getStreamFromPool(false, i));
        }
        return out;
    }
}

#endif /* RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_GET_MASK_H_ */
