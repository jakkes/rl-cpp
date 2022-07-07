#include <gtest/gtest.h>
#include <rl/rl.h>

#include "torch_test.h"


using namespace rl::policies;

TORCH_TEST(policies, sample_gamma, device)
{
    Gamma g{
        torch::tensor({{2.5}, {0.5}}).expand({2, 100000}).clone(),
        5.0 * torch::ones({2, 100000})
    };
    g.to(device);

    auto sample = g.sample();
    ASSERT_EQ(sample.device().type(), device.type());
    auto error = sample.mean(-1).sub(5.0 * torch::tensor({2.5, 0.5}, sample.options())).abs();

    auto e1 = error.index({0}).item().toFloat();
    auto e2 = error.index({1}).item().toFloat();

    ASSERT_LT(e1, 1e-1);
    ASSERT_LT(e2, 1e-1);
}
