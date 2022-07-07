#include <gtest/gtest.h>
#include <torch/torch.h>

#include "rl/policies/dirchlet.h"
#include "torch_test.h"


using namespace rl::policies;

TORCH_TEST(policies, dirchlet, device)
{
    Dirchlet d {
        torch::ones({10, 10, 4})
    };
    d.to(device);

    auto sample = d.sample();
    ASSERT_EQ(sample.device().type(), device.type());
    d.prob(sample);
    d.log_prob(sample);
    d.entropy();

    for (int i = 0; i < 10; i++) {
        sample = d.sample();
        ASSERT_TRUE(
            sample.lt(1).all().item().toBool()
        );
        ASSERT_TRUE(
            sample.gt(0).all().item().toBool()
        );
        ASSERT_TRUE(
            sample.sum(-1).sub_(1.0).lt(1e-6).all().item().toBool()
        );
    } 
}
