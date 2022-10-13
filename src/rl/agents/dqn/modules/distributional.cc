#include "rl/agents/dqn/modules/distributional.h"

#include <exception>

#include <rl/agents/utils/distributional_value.h>


namespace rl::agents::dqn::modules
{
    DistributionalOutput::DistributionalOutput(
        const torch::Tensor &logits,
        const torch::Tensor &atoms
    ) : logits{logits}, atoms{atoms}
    {}

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

        return rl::agents::utils::distributional_value_loss(
            this->logits.index({batchvec, actions}),
            rewards,
            not_terminals,
            next_output_.logits.index({batchvec, next_actions}),
            atoms,
            discount
        );
    }
}
