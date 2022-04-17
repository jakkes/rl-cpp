#include <rl/rl.h>
#include <torch/torch.h>

#include "torch_test.h"


using namespace rl::policies;

TORCH_TEST(policies, factorized, device)
{
    auto o = torch::TensorOptions{}.device(device).dtype(torch::kFloat32);

    auto c1 = std::make_shared<Categorical>(torch::tensor({0.1, 0.9, 0.0}, o));
    auto c2 = std::make_shared<Categorical>(torch::tensor({0.5, 0.0, 0.5}, o));

    Factorized f{ {c1, c2} };

    auto sample = f.sample();

    ASSERT_EQ(sample.size(0), 2);
    
    auto x1 = sample.index({0}).item().toLong();
    ASSERT_TRUE( x1 == 0 || x1 == 1 );
    
    auto x2 = sample.index({1}).item().toLong();
    ASSERT_TRUE( x2 == 0 || x2 == 2 );

    ASSERT_TRUE(
        f.prob(sample).allclose(torch::tensor(0.5 * 0.9, o), 1e-3, 1e-3)
        || f.prob(sample).allclose(torch::tensor(0.5 * 0.1, o), 1e-3, 1e-3)
    );

    f.entropy();
    f.log_prob(sample);
}
