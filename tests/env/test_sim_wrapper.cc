#include <random>

#include <gtest/gtest.h>
#include <rl/rl.h>


using namespace rl;

TEST(test_env, test_sim_wrapper)
{
    std::random_device seeder{};
    auto die = std::bind(
        std::uniform_int_distribution<int>{0, 1},
        std::default_random_engine{seeder()}
    );

    auto sim = std::make_shared<simulators::DiscreteCartPole>(200, 2);
    auto env = std::make_shared<env::SimWrapper>(sim);

    std::shared_ptr<env::State> state = env->reset();
    bool terminal{false};

    while (!terminal)
    {
        auto action = die();

        auto env_obs = env->step(torch::tensor(action));
        auto sim_obs = sim->step(state->state.unsqueeze(0), torch::tensor({action}));

        ASSERT_TRUE(
            (env_obs->state->state == sim_obs.next_states.states.squeeze(0)).all().item().toBool()
        );

        ASSERT_EQ(env_obs->reward, sim_obs.rewards.squeeze(0).item().toFloat());
        ASSERT_EQ(env_obs->terminal, sim_obs.terminals.squeeze(0).item().toBool());

        state = env_obs->state;
        terminal = env_obs->terminal;
    }
}
