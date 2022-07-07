#include <gtest/gtest.h>
#include <torch/torch.h>
#include <rl/rl.h>

#include "torch_test.h"


using namespace rl::policies;

TORCH_TEST(test_policies, test_normal, device)
{
    auto mean = torch::tensor({5.0});
    Normal d{mean, torch::tensor(1.0)};
    d.to(device);

    auto sample = d.sample();
    ASSERT_EQ(sample.device().type(), device.type());

    ASSERT_FLOAT_EQ(d.prob(mean.to(device)).item().toFloat(), M_SQRT1_2 * M_2_SQRTPI / 2);
    ASSERT_FLOAT_EQ(d.log_prob(mean.to(device)).exp().item().toFloat(), M_SQRT1_2 * M_2_SQRTPI / 2);

    auto box = std::make_shared<constraints::Box>(mean.cpu(), torch::ones({1}) + INFINITY);
    box->to(device);
    d.include(box);

    ASSERT_FLOAT_EQ(d.prob(mean.to(device)).item().toFloat(), M_SQRT1_2 * M_2_SQRTPI);
    ASSERT_FLOAT_EQ(d.log_prob(mean.to(device)).exp().item().toFloat(), M_SQRT1_2 * M_2_SQRTPI);

    for (int i = 0; i < 100; i++) {
        auto sample = d.sample();
    }
}
