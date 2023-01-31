#include <gtest/gtest.h>

#include <rl/env/cart_pole.h>
#include <rl/simulators/cart_pole.h>

#include <torch_test.h>

using namespace rl;
using namespace torch::indexing;

TEST(cart_pole, match_env)
{
    env::CartPoleDiscrete env{200, 5};
    simulators::DiscreteCartPole sim{200, 5};

    std::shared_ptr<rl::env::State> env_state = env.reset();
    bool terminal{false};
    int i = 0;

    while (!terminal)
    {
        auto action = torch::tensor(i++ % 5, torch::TensorOptions{}.dtype(torch::kLong));
        auto env_obs = env.step(action);
        auto sim_obs = sim.step(env_state->state.unsqueeze(0), action.unsqueeze(0));
        
        env_state = env_obs->state;
        terminal = env_obs->terminal;

        ASSERT_TRUE(sim_obs.next_states.states.squeeze(0).index({Slice(None, 4)}).allclose(env_state->state.index({Slice(None, 4)})));
        ASSERT_TRUE(sim_obs.terminals.squeeze(0).item().toBool() == terminal);
        ASSERT_TRUE(sim_obs.rewards.squeeze(0).item().toFloat() == env_obs->reward);
    }
}


TORCH_TEST(cart_pole, sparse_reward, device)
{
    simulators::DiscreteCartPole sim{200, 5, simulators::CartPoleOptions{}.sparse_reward_(true).device_(device)};

    bool terminal{false};
    auto state = sim.reset(1);
    int i = 0;
    while (!terminal)
    {
        auto action = torch::tensor(i++ % 5, torch::TensorOptions{}.dtype(torch::kLong).device(device));
        auto sim_obs = sim.step(state.states, action.unsqueeze(0));
        state = sim_obs.next_states;
        terminal = sim_obs.terminals.item().toBool();

        if (sim_obs.terminals.item().toBool()) {
            ASSERT_GT(sim_obs.rewards.item().toFloat(), 8.0f);
        }
        else {
            ASSERT_EQ(sim_obs.rewards.item().toFloat(), 0.0f);
        }
    }
}
