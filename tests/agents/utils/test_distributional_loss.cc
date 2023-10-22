#include <torch/torch.h>
#include <gtest/gtest.h>

#include <torch_test.h>
#include <rl/agents/utils/distributional_loss.h>


torch::Tensor working_distributional_loss(
    const torch::Tensor &current_logits,
    const torch::Tensor &rewards,
    const torch::Tensor &not_terminals,
    const torch::Tensor &next_logits,
    const torch::Tensor &atoms,
    float discount,
    float v_min,
    float v_max,
    bool _
)
{
    auto batchsize = current_logits.size(0);
    auto batchvec = torch::arange(batchsize, torch::TensorOptions{}.device(current_logits.device()));
    auto n_atoms = atoms.size(0);
    auto dz = atoms.index({1}) - atoms.index({0});

    auto m = torch::zeros({batchsize, n_atoms}, rewards.options());

    auto projection = rewards.view({-1, 1}) + not_terminals.view({-1, 1}) * discount * atoms.view({1, -1});
    projection.clamp_(v_min, v_max);
    auto b = (projection - v_min) / dz;

    auto lower = b.floor().to(torch::kLong).clamp_(0, n_atoms - 1);
    auto upper = b.ceil().to(torch::kLong).clamp_(0, n_atoms - 1);

    auto lower_eq_upper = lower == upper;
    if (lower_eq_upper.any().item().toBool()) {
        auto lower_mask = (upper > 0).logical_and_(lower_eq_upper);
        lower.index_put_({lower_mask}, lower.index({lower_mask}) - 1);
    }

    lower_eq_upper = lower == upper;
    if (lower_eq_upper.any().item().toBool()) {
        auto upper_mask = (lower < n_atoms - 1).logical_and_(lower_eq_upper);
        upper.index_put_({upper_mask}, upper.index({upper_mask}) + 1);
    }

    auto index_vec_0 = batchvec.view({-1, 1}).repeat({1, n_atoms}).view({-1});
    auto index_vec_1a = lower.view({-1});


    auto next_distribution = next_logits.softmax(-1);
    m.index_put_(
        {
            index_vec_0,
            lower.view({-1})
        },
        (next_distribution * (upper - b)).view({-1}),
        true
    );

    m.index_put_(
        {
            index_vec_0,
            upper.view({-1})
        },
        (next_distribution * (b - lower)).view({-1}),
        true
    );

    auto log_distribution = torch::log_softmax(current_logits, -1);
    return - (m * log_distribution).sum(-1);
}


TORCH_TEST(distributional_loss, equal, device)
{
    auto current_logits = torch::randn({2048, 21}).to(device);
    auto next_logits = torch::randn({2048, 21}).to(device);

    auto rewards = torch::rand({2048}).to(device);
    auto not_terminals = torch::ones({2048}).to(torch::kBool).to(device);
    auto atoms = torch::linspace(-1.0f, 1.0f, 21).to(device);

    auto output1 = rl::agents::utils::distributional_loss(
        current_logits, rewards, not_terminals, next_logits, atoms, 0.99f, false
    );

    auto output2 = rl::agents::utils::distributional_loss(
        current_logits, rewards, not_terminals, next_logits, atoms, 0.99f, true
    );

    auto output3 = working_distributional_loss(
        current_logits, rewards, not_terminals, next_logits, atoms, 0.99f, -1.0f, 1.0f, true
    );

    ASSERT_TRUE(output1.allclose(output2));
    ASSERT_TRUE(output1.allclose(output3));
}
