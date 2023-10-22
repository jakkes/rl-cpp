#include <gtest/gtest.h>

#include <rl/remote_env/lunar_lander.h>


using namespace rl::remote_env;

TEST(remote, lunar_lander)
{
    LunarLanderFactory factory{"localhost", 50500, 1};
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
        auto observation = env->step(torch::tensor({i % 4}));
    }
}
