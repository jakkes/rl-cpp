#include "rl/agents/alpha_zero/modules/base.h"


namespace rl::agents::alpha_zero::modules
{
    BaseOutput::BaseOutput(const torch::Tensor &prior_probabilities)
        : priors_{prior_probabilities}
    {}


}
