#ifndef RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_GET_MASK_H_
#define RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_GET_MASK_H_


#include <vector>
#include <memory>

#include <torch/torch.h>

#include <rl/policies/constraints/categorical_mask.h>


namespace trainer_impl
{
    inline
    const torch::Tensor get_mask(const rl::policies::constraints::Base &constraint)
    {
        return constraint.as_type<rl::policies::constraints::CategoricalMask>().mask();
    }

    inline
    rl::policies::Categorical mcts_nodes_to_policy(
        const std::vector<std::shared_ptr<MCTSNode>> &nodes,
        const torch::Tensor &masks,
        float temperature
    )
    {
        std::vector<torch::Tensor> visit_counts_vector{}; visit_counts_vector.reserve(nodes.size());
        for (const auto &node : nodes) {
            visit_counts_vector.push_back(node->visit_count());
        }
        auto visit_counts = torch::stack({visit_counts_vector}).to(torch::kFloat32);
        
        visit_counts = visit_counts.index_put_({~masks}, -INFINITY).div_(temperature);

        return rl::policies::Categorical{torch::softmax(visit_counts, -1)};
    }
}

#endif /* RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_GET_MASK_H_ */
