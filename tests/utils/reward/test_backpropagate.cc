#include "torch_test.h"

#include <rl/utils/reward/backpropagate.h>

TORCH_TEST(backpropagate_reward, simple_test, device)
{
    auto rewards = torch::tensor({
        {1.0f, 0.5f, 0.9f},
        {1.0f, 1.0f, 1.0f},
        {1.0f, 0.0f, 0.0f}
    }).to(device);

    auto G = rl::utils::reward::backpropagate(rewards, 0.75f);
    ASSERT_TRUE(G.device().type() == device.type());

    auto G_cpu = G.cpu();
    auto G_accessor = G_cpu.accessor<float, 2>();

    ASSERT_FLOAT_EQ(G_accessor[0][0], 1.0 + 0.75 * 0.5 + 0.75 * 0.75 * 0.9);
    ASSERT_FLOAT_EQ(G_accessor[0][1], 0.5 + 0.75 * 0.9);
    ASSERT_FLOAT_EQ(G_accessor[0][2], 0.9);
    
    ASSERT_FLOAT_EQ(G_accessor[1][0], 1.0 + 0.75 * 1.0 + 0.75 * 0.75 * 1.0);
    ASSERT_FLOAT_EQ(G_accessor[1][1], 1.0 + 0.75 * 1.0);
    ASSERT_FLOAT_EQ(G_accessor[1][2], 1.0);
    
    ASSERT_FLOAT_EQ(G_accessor[2][0], 1.0);
    ASSERT_FLOAT_EQ(G_accessor[2][1], 0.0);
    ASSERT_FLOAT_EQ(G_accessor[2][2], 0.0);
}
