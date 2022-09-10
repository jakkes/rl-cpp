#include <gtest/gtest.h>

#include <rl/utils/n_step_collector.h>


TEST(n_step_collector, loop)
{
    rl::utils::NStepCollector collector{3, 0.75};

    for (int i = 0; i < 3; i++)
    {
        auto state = std::make_shared<rl::env::State>();
        state->state = torch::tensor(i);
        state->action_constraint = nullptr;

        auto action = torch::tensor(10 + i);
        auto reward = 10.0f * i;

        auto out = collector.step(state, action, reward, false);
        ASSERT_EQ(out.size(), 0);
    }

    for (int i = 3; i < 100; i++)
    {
        auto state = std::make_shared<rl::env::State>();
        state->state = torch::tensor(i);
        state->action_constraint = nullptr;

        auto action = torch::tensor(10 + i);
        auto reward = 10.0f * i;

        auto out = collector.step(state, action, reward, false);
        ASSERT_EQ(out.size(), 1);

        ASSERT_NEAR(
            out[0].reward,
            10.0f * ( (i-2) + 0.75 * (i-1) + 0.75 * 0.75 * i ),
            1e-6
        );
        ASSERT_EQ(
            out[0].action.item().toLong(),
            10 + i - 2
        );
        ASSERT_EQ(
            out[0].state->state.item().toLong(),
            i - 2
        );
        ASSERT_EQ(
            out[0].next_state->state.item().toLong(),
            i
        );
        ASSERT_FALSE(out[0].terminal);
    }
}
