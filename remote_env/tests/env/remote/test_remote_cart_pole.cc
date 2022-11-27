#include <gtest/gtest.h>

#include <rl/env/remote/cart_pole.h>


using namespace rl::env::remote;

TEST(remote, cart_pole)
{
    CartPoleFactory factory{"localhost:50051"};
    std::unique_ptr<rl::env::Base> env;

    try {
        env = factory.get();
    } catch (std::runtime_error &e) {
        GTEST_SKIP();
    }

    env->reset();
    auto state = env->state();
    assert(state);

    for (int i = 0; i < 1000; i++) {
        auto observation = env->step(torch::tensor({i % 2}));
        ASSERT_EQ(observation->reward, 1.0f);
    }
}
