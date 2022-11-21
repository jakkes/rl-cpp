#include "rl/agents/alpha_zero/modules/base.h"


namespace rl::agents::alpha_zero::modules
{
    Base::Base(const torch::Tensor &prior_probabilities)
        : priors_{prior_probabilities}
    {}


}
