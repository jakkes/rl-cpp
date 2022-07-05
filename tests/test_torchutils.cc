#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torchdebug.h>
#include <torch_test.h>

#include <rl/torchutils.h>


class TensorHolderImpl : public rl::torchutils::TensorHolder
{
    public:
        torch::Tensor **a, **b;
    
    public:
        TensorHolderImpl(torch::Tensor a, torch::Tensor b)
        : a{register_tensor("a", a)}, b{register_tensor("b", b)}
        {}
};

TORCH_TEST(torchutils, tensor_holder, device)
{
    TensorHolderImpl impl{
        torch::randn({5, 5}),
        torch::randn({3, 3})
    };

    ASSERT_TRUE((*impl.a)->device().is_cpu());
    ASSERT_TRUE((*impl.b)->device().is_cpu());

    impl.to(device);

    ASSERT_EQ((*impl.a)->device().type(), device.type());
    ASSERT_EQ((*impl.b)->device().type(), device.type());
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
