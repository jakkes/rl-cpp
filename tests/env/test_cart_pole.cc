#include <gtest/gtest.h>
#include <torch/torch.h>

#include <rl/env/cart_pole.h>


using namespace rl;

TEST(test_env, test_cart_pole_single_action)
{
    env::CartPoleContinuous env{200};

    env.reset();
    float reward = 0;
    while (!env.is_terminal()) {
        auto obs = env.step(1.0);
        reward += obs->reward;
    }

    ASSERT_LT(reward, 12);
    ASSERT_GE(reward, 8);

    env.reset();
    reward = 0;
    while (!env.is_terminal()) {
        auto obs = env.step(-1.0);
        reward += obs->reward;
    }

    ASSERT_LT(reward, 12);
    ASSERT_GE(reward, 8);
}


TEST(test_env, test_cart_pole_random_action)
{
    float reward = 0;
    env::CartPoleContinuous env{200};

    for (int i = 0; i < 1000; i++) {
        env.reset();
        while (!env.is_terminal()) {
            auto obs = env.step(2 * (torch::rand({}) - 0.5));
            reward += obs->reward;
        }
    }

    ASSERT_LT(reward / 1000, 30);
    ASSERT_GT(reward / 1000, 15);
}
