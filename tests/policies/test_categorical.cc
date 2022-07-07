#include <cmath>

#include <torch/torch.h>
#include <gtest/gtest.h>

#include "rl/policies/policies.h"
#include "torch_test.h"

using namespace rl::policies;

TORCH_TEST(policies, categorical, device)
{
    auto d = Categorical{
        torch::tensor(
            {
                {0.1, 0.5, 0.0, 0.4},
                {0.0, 0.0, 0.0, 1.0}
            }
        )
    };
    d.to(device);

    auto sample = d.sample();
    ASSERT_EQ(sample.device().type(), device.type());

    ASSERT_EQ(sample.size(0), 2);
    ASSERT_NE(sample.index({0}).item().toLong(), 2);
    ASSERT_EQ(sample.index({1}).item().toLong(), 3);

    auto logprob = d.log_prob(sample);
    ASSERT_FLOAT_EQ(
        logprob.index({0}).item().toFloat(),
        std::log(d.get_probabilities().index({0, sample.index({0}).item().toLong()}).item().toFloat())
    );

    ASSERT_FLOAT_EQ(
        logprob.index({1}).item().toFloat(),
        0.0
    );

    auto entropy = d.entropy();
    ASSERT_FLOAT_EQ(entropy.index({0}).item().toFloat(), 0.94334839232f);
    ASSERT_FLOAT_EQ(entropy.index({1}).item().toFloat(), 0.0);
}
