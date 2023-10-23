#include "torch_test.h"

#include <torch/torch.h>
#include <gtest/gtest.h>
#include <rl/agents/dqn/policies/uniform.h>


using namespace rl::agents::dqn::policies;


TORCH_TEST(dqn_policies, uniform_output_probabilities, device)
{
    Uniform policy{};
}
