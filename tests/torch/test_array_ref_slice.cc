#include <rl/torchutils.h>

#include <gtest/gtest.h>
#include <torch/torch.h>


TEST(test_torch, test_array_ref_slice)
{
    auto shape = torch::rand({1, 2, 3, 4, 5}).sizes();
    
    auto x1 = rl::torchutils::slice_shape(shape, 3);
    ASSERT_EQ(x1.size(), 2);
    ASSERT_EQ(x1[0], 4);
    ASSERT_EQ(x1[1], 5);

    auto x2 = rl::torchutils::slice_shape(shape, -2);
    ASSERT_EQ(x2.size(), 3);
    ASSERT_EQ(x2[0], 1);
    ASSERT_EQ(x2[1], 2);
    ASSERT_EQ(x2[2], 3);
}
