#include <torch/torch.h>
#include <gtest/gtest.h>

#include <rl/policies/constraints/constraints.h>
#include <rl/buffers/state_transition.h>
#include <rl/env/observation.h>
#include <rl/env/state.h>

using namespace rl;


void run_state_transition(torch::Device device)
{
    buffers::StateTransition buffer{
        10,
        buffers::StateTransitionOptions{}.device_(device)
    };

    ASSERT_EQ(buffer.size(), 0);
    
    auto state = std::make_shared<env::State>();
    state->action_constraint = std::make_shared<policies::constraints::CategoricalMask>(
        torch::tensor({true, false},
        torch::TensorOptions{}.device(device).dtype(torch::kBool))
    );
    state->state = torch::tensor({0.1, 0.2, 0.3},
                        torch::TensorOptions{}.device(device).dtype(torch::kFloat32));

    auto next_state = std::make_shared<env::State>();
    next_state->action_constraint = std::make_shared<policies::constraints::CategoricalMask>(
        torch::tensor({false, true},
        torch::TensorOptions{}.device(device).dtype(torch::kBool))
    );
    next_state->state = torch::tensor({1.1, 1.2, 1.3},
                        torch::TensorOptions{}.device(device).dtype(torch::kFloat32));

    for (int i = 0; i < 4; i++) buffer.add(state, 1.0, false, next_state);
    ASSERT_EQ(buffer.size(), 4);

    for (int i = 0; i < 10; i++) buffer.add(state, 1.0, false, next_state);
    ASSERT_EQ(buffer.size(), 10);

    auto sample = buffer.sample(100);

    ASSERT_EQ(sample->size(), 100);
    ASSERT_TRUE(
        sample->states->action_constraint->contains(
            torch::zeros({100}, torch::TensorOptions{}.dtype(torch::kLong).device(device))
        ).all().item().toBool()
    );
    ASSERT_FALSE(
        sample->next_states->action_constraint->contains(
            torch::zeros({100}, torch::TensorOptions{}.dtype(torch::kLong).device(device))
        ).any().item().toBool()
    );

    ASSERT_TRUE(sample->next_states->state.sum({-1}).sub_(1.1+1.2+1.3).abs_().lt_(1e-6).all().item().toBool());
    ASSERT_TRUE(sample->states->state.sum({-1}).sub_(0.1+0.2+0.3).abs_().lt_(1e-6).all().item().toBool());
}

TEST(test_buffers, test_state_transition_cpu)
{
    run_state_transition(torch::kCPU);
}

TEST(test_buffers, test_state_transition_cuda)
{
    if (!torch::cuda::is_available()) GTEST_SKIP();
    run_state_transition(torch::kCUDA);
}
