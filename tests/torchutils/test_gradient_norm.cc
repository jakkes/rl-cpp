#include <math.h>
#include <torch_test.h>
#include <rl/torchutils/gradient_norm.h>
#include <torch/torch.h>


TORCH_TEST(gradient_norm, gradient_norm, device)
{
    auto w = torch::tensor({1.0, 2.0}, torch::TensorOptions{}.device(device));
    w.set_requires_grad(true);

    auto optimizer = std::make_shared<torch::optim::SGD>(
        std::vector{w},
        torch::optim::SGDOptions{1.0}
            .nesterov(false)
            .momentum(0.0)
    );

    auto x = torch::tensor({2.0, 3.0}, torch::TensorOptions{}.device(device));

    auto loss = (w * x).sum().square();
    loss.backward();

    ASSERT_FLOAT_EQ(w.grad().index({0}).item().toFloat(), 32.0f);
    ASSERT_FLOAT_EQ(w.grad().index({1}).item().toFloat(), 48.0f);

    ASSERT_FLOAT_EQ(rl::torchutils::compute_gradient_norm(optimizer).item().toFloat(), std::sqrt(32.0 * 32.0 + 48.0 * 48.0));
}
