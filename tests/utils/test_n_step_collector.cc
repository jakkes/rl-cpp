#include <gtest/gtest.h>

#include <rl/utils/n_step_collector.h>


void check_transition(
    const rl::utils::NStepCollectorTransition &transition,
    int64_t start_state,
    int64_t action,
    float reward,
    bool terminal,
    int64_t end_state
)
{
    ASSERT_NEAR(
        transition.reward,
        reward,
        1e-6
    );
    ASSERT_EQ(
        transition.action.item().toLong(),
        action
    );
    ASSERT_EQ(
        transition.state->state.item().toLong(),
        start_state
    );
    ASSERT_EQ(
        transition.next_state->state.item().toLong(),
        end_state
    );
    ASSERT_EQ(transition.terminal, terminal);
}

auto step(rl::utils::NStepCollector *collector, int i, bool terminal)
{
    auto state = std::make_shared<rl::env::State>();
    state->state = torch::tensor(i);
    state->action_constraint = nullptr;

    auto action = torch::tensor(10 + i);
    auto reward = 10.0f * i;

    return collector->step(state, action, reward, terminal);
}


TEST(n_step_collector, terminal_single_step)
{
    rl::utils::NStepCollector collector{3, 0.75};

    auto o = std::vector{
        step(&collector, 0, true),
        step(&collector, 1, true),
        step(&collector, 2, true),
        step(&collector, 3, true)
    };

    for (int i = 0; i < 4; i++)
    {
        ASSERT_EQ(o[i].size(), 1);

        check_transition(
            o[i][0],
            i,
            10 + i,
            10.0f * i,
            true,
            i
        );
    }
}


TEST(n_step_collector, loop)
{
    rl::utils::NStepCollector collector{3, 0.75};

    for (int i = 0; i < 3; i++)
    {
        auto out = step(&collector, i, false);
        ASSERT_EQ(out.size(), 0);
    }

    for (int i = 3; i < 100; i++)
    {
        auto out = step(&collector, i, false);
        ASSERT_EQ(out.size(), 1);

        check_transition(
            out[0],
            i-3,
            7 + i,
            10.0f * ( (i-3) + 0.75 * (i-2) + 0.75 * 0.75 * (i-1) ),
            false,
            i
        );
    }
}


TEST(n_step_collector, loop_terminal)
{
    rl::utils::NStepCollector collector{3, 0.75};

    auto step = [&collector] (int i, bool terminal)
    {
        auto state = std::make_shared<rl::env::State>();
        state->state = torch::tensor(i);
        state->action_constraint = nullptr;

        auto action = torch::tensor(10 + i);
        auto reward = 10.0f * i;

        return collector.step(state, action, reward, terminal);
    };

    for (int i = 0; i < 10; i++) {
        step(i, false);
    }

    auto out = step(10, true);

    ASSERT_EQ(out.size(), 4);

    std::unordered_set<float> rewards {
        10.0f * ( 7 + 0.75 * 8 + 0.75 * 0.75 * 9 ),
        10.0f * ( 8 + 0.75 * 9 + 0.75 * 0.75 * 10),
        10.0f * ( 9 + 0.75 * 10),
        10.0f * ( 10 )
    };
    std::unordered_set<int64_t> actions { 17, 18, 19, 20 };
    std::unordered_set<int64_t> states { 7, 8, 9, 10 };
    int n_terminals = 0;

    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE( rewards.find(out[i].reward) != rewards.end() );
        rewards.erase(out[i].reward);

        ASSERT_TRUE( actions.find(out[i].action.item().toLong()) != actions.end() );
        actions.erase( out[i].action.item().toLong() );

        ASSERT_TRUE( states.find(out[i].state->state.item().toLong()) != states.end() );
        states.erase( out[i].state->state.item().toLong() );

        ASSERT_TRUE( out[i].next_state->state.item().toLong() == 10 );
        if (out[i].terminal) n_terminals++;
    }
    ASSERT_EQ(n_terminals, 3);

    for (int i = 0; i < 3; i++)
    {
        auto out = step(11 + i, false);
        ASSERT_EQ(out.size(), 0);
    }

    for (int i = 0; i < 10; i++) {
        auto out = step(14 + i, false);
        ASSERT_EQ(out.size(), 1);
        check_transition(
            out[0],
            11 + i,
            10 + 11 + i,
            10.0f * ( (11+i) + 0.75 * (12+i) + 0.75 * 0.75 * (13+i) ),
            false,
            14+i
        );
    }
}


// TEST(n_step_collector, no_loop_terminal)
// {
//     rl::utils::NStepCollector collector{3, 0.75};

//     auto step = [&collector] (int i, bool terminal)
//     {
//         auto state = std::make_shared<rl::env::State>();
//         state->state = torch::tensor(i);
//         state->action_constraint = nullptr;

//         auto action = torch::tensor(10 + i);
//         auto reward = 10.0f * i;

//         return collector.step(state, action, reward, terminal);
//     };

    
// }
