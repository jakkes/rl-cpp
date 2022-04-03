#include <gtest/gtest.h>
#include <rl/rl.h>

#include "torch_test.h"


using namespace rl::policies;

TORCH_TEST(policies, sample_gamma, device)
{
    auto o = torch::TensorOptions{}.device(device).dtype(torch::kFloat32);

    Gamma g{
        torch::tensor({0.5, 1.0, 2.0}, o),
        torch::ones({3}, o)
    };

    auto sample = g.sample();
}
