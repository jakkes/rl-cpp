#include <gtest/gtest.h>
#include <torch/torch.h>

#include <rl/policies/policies.h>


using namespace rl::policies::constraints;
using namespace rl::policies;


void run_categorical_mask(torch::Device device)
{
    auto c = std::make_shared<CategoricalMask>(
        torch::tensor({true, true, false}, torch::TensorOptions{}.dtype(torch::kBool).device(device))
    );

    ASSERT_TRUE(c->contains(torch::tensor(0, torch::TensorOptions{}.dtype(torch::kLong).device(device))).item().toBool());
    ASSERT_TRUE(c->contains(torch::tensor(1, torch::TensorOptions{}.dtype(torch::kLong).device(device))).item().toBool());
    ASSERT_FALSE(c->contains(torch::tensor(2, torch::TensorOptions{}.dtype(torch::kLong).device(device))).item().toBool());

    auto r = c->contains(torch::tensor({{0, 1}, {1, 2}, {2, 2}}, torch::TensorOptions{}.dtype(torch::kLong).device(device)));

    ASSERT_EQ(r.size(0), 3);
    ASSERT_EQ(r.size(1), 2);

    ASSERT_TRUE(r.index({0, 0}).item().toBool());
    ASSERT_TRUE(r.index({0, 1}).item().toBool());
    ASSERT_TRUE(r.index({1, 0}).item().toBool());
    ASSERT_FALSE(r.index({1, 1}).item().toBool());
    ASSERT_FALSE(r.index({2, 0}).item().toBool());
    ASSERT_FALSE(r.index({2, 1}).item().toBool());

    auto stacked_c = stack<CategoricalMask>({c, c});
    auto stacked_r = stacked_c->contains(
        torch::tensor({0, 2}, torch::TensorOptions{}.dtype(torch::kLong).device(device))
    );
    ASSERT_TRUE(stacked_r.index({0}).item().toBool());
    ASSERT_FALSE(stacked_r.index({1}).item().toBool());
}

TEST(test_policy_constraints, categorical_mask_cpu)
{
    run_categorical_mask(torch::kCPU);
}

TEST(test_policy_constraints, categorical_mask_gpu)
{
    if (!torch::cuda::is_available()) GTEST_SKIP();
    run_categorical_mask(torch::kCUDA);
}


void run_categorical_mask_batch(torch::Device device)
{
    auto c = std::make_shared<CategoricalMask>(
        torch::tensor(
            {
                {true, true, false},
                {false, false, true}
            },
            torch::TensorOptions{}.dtype(torch::kBool).device(device)
        )
    );

    auto r1 = c->contains(torch::tensor({0, 0}, torch::TensorOptions{}.dtype(torch::kLong).device(device)));
    auto r2 = c->contains(torch::tensor({1, 1}, torch::TensorOptions{}.dtype(torch::kLong).device(device)));
    auto r3 = c->contains(torch::tensor({2, 2}, torch::TensorOptions{}.dtype(torch::kLong).device(device)));

    ASSERT_TRUE(r1.index({0}).item().toBool());
    ASSERT_FALSE(r1.index({1}).item().toBool());

    ASSERT_TRUE(r2.index({0}).item().toBool());
    ASSERT_FALSE(r2.index({1}).item().toBool());

    ASSERT_FALSE(r3.index({0}).item().toBool());
    ASSERT_TRUE(r3.index({1}).item().toBool());
}

TEST(test_policy_constraints, categorical_mask_batch_cpu)
{
    run_categorical_mask_batch(torch::kCPU);
}

TEST(test_policy_constraints, categorical_mask_batch_gpu)
{
    if (!torch::cuda::is_available()) GTEST_SKIP();
    run_categorical_mask_batch(torch::kCUDA);
}


void run_categorical_mask_on_categorical_policy(torch::Device device)
{
    auto d = Categorical{
        torch::tensor(
            {
                {0.1, 0.5, 10.0, 0.4},
                {1.0, 1.0, 1.0, 1.0}
            },
            torch::TensorOptions{}.device(device)
        )
    };
    auto c = std::make_shared<CategoricalMask>(
        torch::tensor(
            {
                {true, true, false, true},
                {false, false, false, true}
            },
            torch::TensorOptions{}.dtype(torch::kBool).device(device)
        )
    );
    d.include(c);

    auto sample = d.sample();
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
}

TEST(test_policy_constraints, categorical_mask_on_categorical_policy_cpu)
{
    run_categorical_mask_on_categorical_policy(torch::kCPU);
}

TEST(test_policy_constraints, categorical_mask_on_categorical_policy_cuda)
{
    run_categorical_mask_on_categorical_policy(torch::kCUDA);
}

