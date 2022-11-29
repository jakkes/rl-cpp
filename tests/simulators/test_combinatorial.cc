#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <rl/simulators/combinatorial_lock.h>


using namespace rl::simulators;

TEST(combinatorial, short_sequence)
{
    std::vector correct_sequence{0, 3, 2, 1, 2};
    int dim{4};
    int n{5};

    CombinatorialLock sim{dim, correct_sequence};

    Observations obs{};
    obs.next_states = sim.reset(n);
    for (int i = 0; i < correct_sequence.size(); i++)
    {
        auto actions = torch::tensor({i % dim, (i+1) % dim, correct_sequence[i], (i+2) % dim, (i+5) % dim});
        obs = sim.step(obs.next_states.states, actions);

        if (i < correct_sequence.size() - 1) {
            ASSERT_FALSE(obs.terminals.any().item().toBool());
            ASSERT_TRUE((obs.rewards == 0.0f).all().item().toBool());
        }
    }

    for (int i = 0; i < n; i++)
    {
        ASSERT_TRUE(obs.terminals.index({i}).item().toBool());
        ASSERT_EQ(obs.rewards.index({i}).item().toFloat(), i == 2 ? 1.0f : 0.0f);
    }
}
