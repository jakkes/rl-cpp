#include <gtest/gtest.h>

#include <torch/torch.h>
#include <rl/policies/constraints/box.h>
#include <torch_test.h>


using namespace rl::policies::constraints;
using namespace torch::indexing;

TORCH_TEST(test_policy_constraints, test_box, device)
{
    Box box{
        -1.0 * torch::ones({2}),
        torch::ones({2}),
        BoxOptions{}
            .inclusive_lower_(true)
            .inclusive_upper_(false)
            .n_action_dims_(1)
    };
    box.to(device);
    auto x1 = torch::linspace(-1.0, 1.0, 100);
    auto x2 = torch::zeros({100});
    auto x = torch::stack({x1, x2}, 1).to(device);

    auto y = box.contains(x);

    ASSERT_EQ(y.device().type(), device.type());
    ASSERT_TRUE(y.index({Slice(0, -1)}).all().item().toBool());
    ASSERT_FALSE(y.index({-1}).item().toBool());
}
