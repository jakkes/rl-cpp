#include <gtest/gtest.h>

#include <torch/torch.h>
#include <rl/policies/constraints/box.h>


using namespace rl::policies::constraints;
using namespace torch::indexing;

TEST(test_policy_constraints, test_box)
{
    Box box{
        -1.0 * torch::ones({2}),
        torch::ones({2}),
        BoxOptions{}
            .inclusive_lower_(true)
            .inclusive_upper_(false)
            .n_action_dims_(1)
    };
    auto x1 = torch::linspace(-1.0, 1.0, 100);
    auto x2 = torch::zeros({100});
    auto x = torch::stack({x1, x2}, 1);

    auto y = box.contains(x);

    ASSERT_TRUE(y.index({Slice(0, -1)}).all().item().toBool());
    ASSERT_FALSE(y.index({-1}).item().toBool());
}
