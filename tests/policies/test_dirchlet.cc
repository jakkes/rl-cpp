#include <gtest/gtest.h>
#include <torch/torch.h>

#include "rl/policies/dirchlet.h"
#include "torch_test.h"


using namespace rl::policies;

TORCH_TEST(policies, dirchlet, device)
{
    auto o = torch::TensorOptions{}.device(device);

    Dirchlet d {
        torch::ones({10, 10, 4}, o),
        - 5 + torch::zeros({10, 10, 4}, o),
        5 + torch::zeros({10, 10, 4}, o)
    };

    auto sample = d.sample();
    d.prob(sample);
    d.log_prob(sample);
    d.entropy();

    for (int i = 0; i < 1000; i++) {
        sample = d.sample();
        ASSERT_TRUE(
            sample.lt(5).all().item().toBool()
        );
        ASSERT_TRUE(
            sample.gt(-5).all().item().toBool()
        );
        std::cout << sample.abs_().sum(-1).max() << "\n";
        ASSERT_TRUE(
            sample.abs_().sum(-1).lt(10).all().item().toBool()
        );
    } 
}
