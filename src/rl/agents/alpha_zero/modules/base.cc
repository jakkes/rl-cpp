#include "rl/agents/alpha_zero/modules/base.h"


namespace rl::agents::alpha_zero::modules
{
    BaseOutput::BaseOutput(const torch::Tensor &probabilities)
    : policy_{probabilities}
    {}
}
