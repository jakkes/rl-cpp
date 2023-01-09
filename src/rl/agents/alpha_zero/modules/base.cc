#include "rl/agents/alpha_zero/modules/base.h"


namespace rl::agents::alpha_zero::modules
{
    BaseOutput::BaseOutput(const torch::Tensor &policy_logits)
    : policy_{torch::softmax(policy_logits.detach(), -1)}, policy_logits{policy_logits}
    {}
}
