#include "rl/agents/dqn/value_parsers/distributional.h"

#include <rl/agents/utils/distributional_loss.h>


namespace rl::agents::dqn::value_parsers
{
    Distributional::Distributional(const torch::Tensor &atoms, bool cuda_graph_compatible)
        : atoms(atoms), cuda_graph_compatible(cuda_graph_compatible)
    {
        assert(atoms.dim() == 1);
    }

    torch::Tensor Distributional::values(
        const torch::Tensor &module_output, const torch::Tensor &mask
    ) const
    {
        auto values = (atoms * torch::softmax(module_output, -1)).sum(-1);
        return values.where(mask, torch::zeros_like(values.detach()) - INFINITY);
    }

    torch::Tensor Distributional::loss(
        const torch::Tensor &module_output,
        const torch::Tensor &masks,
        const torch::Tensor &actions,
        const torch::Tensor &rewards,
        const torch::Tensor &not_terminals,
        const torch::Tensor &next_module_output,
        const torch::Tensor &next_masks,
        const torch::Tensor &next_actions,
        float discount
    ) const
    {
        auto batchvec = torch::arange(
            actions.size(0), torch::TensorOptions{}.device(actions.device())
        );
        
        return rl::agents::utils::distributional_loss(
            module_output.index({batchvec, actions}),
            rewards,
            not_terminals,
            next_module_output.index({batchvec, next_actions}),
            atoms,
            discount,
            cuda_graph_compatible
        );
    }
}
