#include <torch/torch.h>
#include <gtest/gtest.h>

#include <torch_test.h>
#include <rl/agents/utils/distributional_loss.h>


TORCH_TEST(distributional_loss, equal, device)
{
    auto current_logits = torch::randn({128, 10}).to(device);
    auto next_logits = torch::randn({128, 10}).to(device);

    auto rewards = torch::rand({128}).to(device);
    auto not_terminals = torch::ones({128}).to(torch::kBool).to(device);
    auto atoms = torch::linspace(0.0f, 1.0f, 10).to(device);

    auto output1 = rl::agents::utils::distributional_loss(
        current_logits, rewards, not_terminals, next_logits, atoms, 1.0f, 0.0f, 1.0f, false
    );

    auto output2 = rl::agents::utils::distributional_loss(
        current_logits, rewards, not_terminals, next_logits, atoms, 1.0f, 0.0f, 1.0f, true
    );

    ASSERT_TRUE(output1.allclose(output2));
}
