#include "torch_test.h"

#include <rl/agents/dqn/value_parsers/estimated_mean.h>
#include <rl/policies/constraints/categorical_mask.h>

using namespace rl::agents::dqn::value_parsers;

TORCH_TEST(dqn_value_parsers, estimated_mean_loss, device)
{
    EstimatedMean value_parser{};

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

    auto rewards = torch::tensor({0.5, 0.2, 0.1, 0.9},
                            torch::TensorOptions{}.device(device));

    auto not_terminals = torch::tensor({true, true, false, true},
                            torch::TensorOptions{}.device(device).dtype(torch::kBool));

    // Take disallowed action on purpose.
    auto actions = torch::tensor({1, 0, 1, 2},
                            torch::TensorOptions{}.device(device).dtype(torch::kLong));
    auto next_actions = torch::tensor({1, 0, 1, 1},
                            torch::TensorOptions{}.device(device).dtype(torch::kLong));

    float discount = 0.75;

    auto loss = value_parser.loss(
        values,
        masks,
        actions,
        rewards,
        not_terminals,
        next_values,
        next_masks,
        next_actions,
        discount
    );

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


TORCH_TEST(dqn_value_parsers, estimated_mean_values, device)
{
    EstimatedMean value_parser{};

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

    auto output_values = value_parser.values(values, masks);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == 0 && j == 1) {
                ASSERT_TRUE(output_values.index({i, j}).isneginf().item().toBool());
                continue;
            }

            ASSERT_NEAR(
                output_values.index({i, j}).item().toFloat(),
                values.index({i, j}).item().toFloat(),
                1e-16
            );
        }
    }
}
