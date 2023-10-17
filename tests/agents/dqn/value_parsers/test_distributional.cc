#include "torch_test.h"

#include <torch/torch.h>
#include <gtest/gtest.h>
#include <rl/agents/dqn/value_parsers/distributional.h>


using namespace rl::agents::dqn::value_parsers;

TORCH_TEST(dqn_value_parsers, distributional_output_value, device)
{
    auto atoms = torch::linspace(0.0, 1.0, 3).to(device);
    Distributional value_parser(atoms);

    auto distributions = torch::tensor({
        {
            {0.2f, 0.6f, 0.2f},
            {0.0f, 0.0f, 1.0f},
            {1.0f, 0.0f, 0.0f}
        },
        {
            {0.5f, 0.1f, 0.4f},
            {0.0f, 1.0f, 0.0f},
            {1.0f, 0.0f, 0.0f}
        }
    }).to(device);

    auto logits = torch::log(distributions * distributions.exp().sum(-1, true));

    auto masks = torch::tensor({
        {true, true, false},
        {true, false, true}
    }).to(device);

    auto values = value_parser.values(logits, masks);
    
    ASSERT_NEAR(
        values.index({0, 0}).item().toFloat(),
        0.6 * 0.5 + 0.2,
        1e-6
    );

    ASSERT_NEAR(
        values.index({0, 1}).item().toFloat(),
        1.0,
        1e-6
    );

    ASSERT_TRUE(
        values.index({0, 2}).isneginf().item().toBool()
    );

    ASSERT_NEAR(
        values.index({1, 0}).item().toFloat(),
        0.5 * 0.1 + 0.4 * 1.0,
        1e-6
    );

    ASSERT_TRUE(
        values.index({1, 1}).isneginf().item().toBool()
    );

    ASSERT_NEAR(
        values.index({1, 2}).item().toFloat(),
        0.0,
        1e-6
    );
}
