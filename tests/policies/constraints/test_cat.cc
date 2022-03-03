#include <memory>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <rl/policies/constraints/constraints.h>

using namespace rl::policies::constraints;


void run_concat(torch::Device device)
{
    std::shared_ptr<Base> c1 = std::make_shared<CategoricalMask>(
        torch::tensor({true, true, false}, torch::TensorOptions{}.dtype(torch::kBool).device(device))
    );

    std::shared_ptr<Base> c2 = std::make_shared<CategoricalMask>(
        torch::tensor({false, true, true}, torch::TensorOptions{}.dtype(torch::kBool).device(device))
    );
    std::vector<std::shared_ptr<Base>> c1c2{c1, c2};

    auto cat = std::make_shared<Concat>(c1c2);

    ASSERT_FALSE(cat->contains(torch::tensor(0, torch::TensorOptions{}.dtype(torch::kLong).device(device))).item().toBool());
    ASSERT_TRUE(cat->contains(torch::tensor(1, torch::TensorOptions{}.dtype(torch::kLong).device(device))).item().toBool());
    ASSERT_FALSE(cat->contains(torch::tensor(2, torch::TensorOptions{}.dtype(torch::kLong).device(device))).item().toBool());

    auto cat2 = std::make_shared<Concat>(c1c2);
}

TEST(test_policy_constraints, concat_cpu)
{
    run_concat(torch::kCPU);
}

TEST(test_policy_constraints, concat_gpu)
{
    if (!torch::cuda::is_available()) GTEST_SKIP();
    run_concat(torch::kCUDA);
}
