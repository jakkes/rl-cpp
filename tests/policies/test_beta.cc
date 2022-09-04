#include <gtest/gtest.h>

#include <rl/policies/beta.h>
#include "torch_test.h"

using namespace rl::policies;


TORCH_TEST(policies, beta, device)
{
    Beta d{
        torch::tensor({0.1, 1.0, 2.0}),
        torch::tensor({0.5, 1.0, 2.0}),
        -1 * torch::ones({3}),
        torch::ones({3})
    };

    d.to(device);

    for (int i = 0; i < 100; i++) {
        auto sample = d.sample();
        ASSERT_TRUE((sample >= -1).logical_and_(sample <= 1).all().item().toBool());
        ASSERT_EQ(sample.size(0), 3);
    }

    auto sample = d.sample();
    ASSERT_EQ(sample.device().type(), device.type());

    d.prob(sample);
    d.log_prob(sample);
    d.entropy();
}


TORCH_TEST(policies, beta_mean, device)
{
    auto alpha = torch::tensor({0.1, 0.9, 2.0});
    auto beta = torch::tensor({0.5, 1.1, 0.9});
    auto a = torch::tensor({-5.0, 0.0, 2.0});
    auto b = torch::tensor({-3.0, 1.0, 2.1});

    Beta d{
        alpha.unsqueeze(1).repeat({1, 100000}),
        beta.unsqueeze(1).repeat({1, 100000}),
        a.unsqueeze(1).repeat({1, 100000}),
        b.unsqueeze(1).repeat({1, 100000})
    };
    d.to(device);

    auto sample = d.sample();
    ASSERT_EQ(sample.device().type(), device.type());
    auto mean = sample.mean(1).cpu();

    auto expected_mean = 1.0 / (1.0 + beta / alpha) * (b - a) + a;

    ASSERT_TRUE(mean.allclose(expected_mean, 1e-2, 1e-2));
}

TORCH_TEST(policies, sample_beta_lower_limit, device)
{
    Beta d{
        torch::tensor({0.1f}).repeat({10000}),
        torch::tensor({0.1f}).repeat({10000})
    };
    d.to(device);

    for (int i = 0; i < 100; i++) {
        auto sample = d.sample();
        ASSERT_FALSE(sample.isnan().any().item().toBool());
    }
}
