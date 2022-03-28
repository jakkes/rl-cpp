#include <gtest/gtest.h>

#include <rl/policies/beta.h>

using namespace rl::policies;


void run_beta(torch::Device device)
{
    Beta d{
        torch::tensor({0.1, 1.0, 2.0}, torch::TensorOptions{}.device(device)),
        torch::tensor({0.5, 1.0, 2.0}, torch::TensorOptions{}.device(device)),
        -1 * torch::ones({3}, torch::TensorOptions{}.device(device)),
        torch::ones({3}, torch::TensorOptions{}.device(device))
    };

    for (int i = 0; i < 100; i++) {
        auto sample = d.sample();
        ASSERT_TRUE((sample > -1).logical_and_(sample < 1).all().item().toBool());
        ASSERT_EQ(sample.size(0), 3);
    }

    auto sample = d.sample();

    d.prob(sample);
    d.log_prob(sample);
    d.entropy();
}


TEST(test_policies, test_beta_cpu)
{
    run_beta(torch::kCPU);
}

TEST(test_policies, test_beta_cuda)
{
    if (!torch::cuda::is_available()) GTEST_SKIP();
    run_beta(torch::kCUDA);
}
