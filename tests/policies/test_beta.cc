#include <gtest/gtest.h>

#include <rl/policies/beta.h>
#include "torch_test.h"

using namespace rl::policies;


TORCH_TEST(policies, beta, device)
{
    auto o = torch::TensorOptions{}.device(device);
    Beta d{
        torch::tensor({0.1, 1.0, 2.0}, o),
        torch::tensor({0.5, 1.0, 2.0}, o),
        -1 * torch::ones({3}, o),
        torch::ones({3}, o)
    };

    for (int i = 0; i < 100; i++) {
        auto sample = d.sample();
        ASSERT_TRUE((sample >= -1).logical_and_(sample <= 1).all().item().toBool());
        ASSERT_EQ(sample.size(0), 3);
    }

    auto sample = d.sample();

    d.prob(sample);
    d.log_prob(sample);
    d.entropy();
}


TORCH_TEST(policies, beta_mean, device)
{
    auto o = torch::TensorOptions{}.device(device);
    auto alpha = torch::tensor({0.1, 0.9, 2.0}, o);
    auto beta = torch::tensor({0.5, 1.1, 0.9}, o);
    auto a = torch::tensor({-5.0, 0.0, 2.0}, o);
    auto b = torch::tensor({-3.0, 1.0, 2.1}, o);

    Beta d{
        alpha.unsqueeze(1).repeat({1, 100000}),
        beta.unsqueeze(1).repeat({1, 100000}),
        a.unsqueeze(1).repeat({1, 100000}),
        b.unsqueeze(1).repeat({1, 100000})
    };

    auto sample = d.sample();
    auto mean = sample.mean(1);

    auto expected_mean = 1.0 / (1.0 + beta / alpha) * (b - a) + a;

    ASSERT_TRUE(mean.allclose(expected_mean, 1e-2, 1e-2));
}
