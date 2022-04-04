#include <gtest/gtest.h>
#include <rl/rl.h>

#include "torch_test.h"


using namespace rl::policies;

TORCH_TEST(policies, sample_gamma, device)
{
    auto o = torch::TensorOptions{}.device(device).dtype(torch::kFloat32);

    Gamma g{
        torch::tensor({{2.5}, {0.5}}, o).expand({2, 100000}).clone(),
        5.0 * torch::ones({2, 100000}, o)
    };

    auto sample = g.sample();
    auto error = sample.mean(-1).sub(5.0 * torch::tensor({2.5, 0.5}, o)).abs();

    auto e1 = error.index({0}).item().toFloat();
    auto e2 = error.index({1}).item().toFloat();

    ASSERT_LT(e1, 1e-1);
    ASSERT_LT(e2, 1e-1);
}
