#include <vector>

#include <gtest/gtest.h>

#include "rl/cpputils/slice_vector.h"


TEST(test_cpputils, vec_slice)
{
    std::vector<int> x{0, 1, 2, 3, 4};
    
    auto x1 = rl::cpputils::slice(x, 1, -1);
    ASSERT_EQ(x1.size(), 3);
    ASSERT_EQ(x1[0], 1);
    ASSERT_EQ(x1[1], 2);
    ASSERT_EQ(x1[2], 3);

    auto x2 = rl::cpputils::slice(x, 0, -1);
    ASSERT_EQ(x2.size(), 4);
    ASSERT_EQ(x2[0], 0);
    ASSERT_EQ(x2[1], 1);
    ASSERT_EQ(x2[2], 2);
    ASSERT_EQ(x2[3], 3);
}
