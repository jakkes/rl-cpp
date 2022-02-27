#include <memory>

#include <torch/torch.h>
#include <gtest/gtest.h>

#include "rl/policies/policies.h"

using namespace rl::policies;

void run_rejection_sampler(torch::Device device)
{
    auto d = std::make_unique<Categorical>(
        torch::tensor(
            {0.1, 0.5, 0.0, 0.4},
            torch::TensorOptions{}.device(device)
        )
    );

    auto c1 = std::make_shared<constraints::CategoricalMask>(
        torch::tensor({true, true, true, false},
        torch::TensorOptions{}.dtype(torch::kBool).device(device))
    );
    auto c2 = std::make_shared<constraints::CategoricalMask>(
        torch::tensor({true, false, true, true},
        torch::TensorOptions{}.dtype(torch::kBool).device(device))
    );

    auto rejsampler = RejectionSampler{
        std::move(d),
        {c1, c2}
    };

    auto sample = rejsampler.sample();
    ASSERT_EQ(sample.item().toLong(), 0);
}

TEST(test_policies, rejection_sampler_cpu)
{
    run_rejection_sampler(torch::kCPU);
}

TEST(test_policies, rejection_sampler_cuda)
{
    if (!torch::cuda::is_available()) GTEST_SKIP();
    run_rejection_sampler(torch::kCUDA);
}
