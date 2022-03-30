#include <gtest/gtest.h>
#include <torch/torch.h>
#include <rl/rl.h>


using namespace rl::policies;

TEST(test_policies, test_normal)
{
    auto mean = torch::tensor({5.0});
    Normal d{mean, torch::tensor(1.0)};

    ASSERT_FLOAT_EQ(d.prob(mean).item().toFloat(), M_SQRT1_2 * M_2_SQRTPI / 2);
    ASSERT_FLOAT_EQ(d.log_prob(mean).exp().item().toFloat(), M_SQRT1_2 * M_2_SQRTPI / 2);

    auto box = std::make_shared<constraints::Box>(mean, torch::ones({1}) + INFINITY);
    d.include(box);

    ASSERT_FLOAT_EQ(d.prob(mean).item().toFloat(), M_SQRT1_2 * M_2_SQRTPI);
    ASSERT_FLOAT_EQ(d.log_prob(mean).exp().item().toFloat(), M_SQRT1_2 * M_2_SQRTPI);

    for (int i = 0; i < 100; i++) {
        auto sample = d.sample();
    }
}
