#include "torch_test.h"

#include <rl/agents/dqn/modules/dqn.h>
#include <rl/policies/constraints/categorical_mask.h>

using namespace rl::agents::dqn::modules;

TORCH_TEST(dqn_module, dqn_output_loss, device)
{
    auto values = torch::tensor({
        {0.5, 1.0, 0.0},
        {0.5, 1.0, 0.0},
        {0.5, 1.0, 0.0},
        {0.5, 1.0, 0.0}
    }, torch::TensorOptions{}.device(device).requires_grad(true));
    auto masks = torch::tensor({
        {true, false, true},
        {true, true, true},
        {true, true, true},
        {true, true, true}
    }, torch::TensorOptions{}.device(device).dtype(torch::kBool));
    DQNOutput output{values};
    output.apply_mask(masks);

    auto next_values = torch::tensor({
        {0.5, 1.0, 0.0},
        {0.5, 1.0, 0.0},
        {0.5, 1.0, 0.0},
        {0.5, 1.0, 0.0}
    }, torch::TensorOptions{}.device(device));
    auto next_masks = torch::tensor({
        {true, true, true},
        {true, false, true},
        {true, true, true},
        {true, true, true}
    }, torch::TensorOptions{}.device(device).dtype(torch::kBool));

    DQNOutput next_output{next_values};
    next_output.apply_mask(next_masks);

    auto rewards = torch::tensor({0.5, 0.2, 0.1, 0.9},
                            torch::TensorOptions{}.device(device));

    auto not_terminals = torch::tensor({true, true, false, true},
                            torch::TensorOptions{}.device(device).dtype(torch::kBool));

    auto actions = torch::tensor({1, 0, 1, 2},
                            torch::TensorOptions{}.device(device).dtype(torch::kLong));
        
    float discount = 0.75;

    auto loss = output.loss(actions, rewards, not_terminals, next_output, next_output.greedy_action(), discount);

    ASSERT_TRUE(
        loss.index({0}).isinf().item().toBool()
    );
    ASSERT_NEAR(
        loss.index({1}).item().toFloat(),
        0.005625,
        1e-6
    );
    ASSERT_NEAR(
        loss.index({2}).item().toFloat(),
        0.81,
        1e-6
    );
    ASSERT_NEAR(
        loss.index({3}).item().toFloat(),
        2.7225,
        1e-6
    );
}
