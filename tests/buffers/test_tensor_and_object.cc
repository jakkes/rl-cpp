#include <torch/torch.h>
#include <gtest/gtest.h>

#include "rl/rl.h"
#include "torch_test.h"

using namespace rl;


struct P {
    int x;
    int y;

    P() {};
    P(int x, int y) : x{x}, y{y} {}
};


TORCH_TEST(buffers, tensor_and_object, device)
{
    std::vector<int64_t> shape1 = {1, 2, 3};
    auto options1 = torch::TensorOptions{}.dtype(torch::kFloat32).device(device);

    std::vector<int64_t> shape2 = {4, 5, 6};
    auto options2 = torch::TensorOptions{}.dtype(torch::kBool).device(device);

    auto shapes = {shape1, shape2};
    auto options = {options1, options2};
    
    auto buffer = std::make_shared<buffers::TensorAndObject<P>>(10, shapes, options);
    auto sampler = buffers::samplers::Uniform<buffers::TensorAndObject<P>>(buffer);

    ASSERT_EQ(buffer->size(), 0);

    auto x = P(1, 2);
    auto y = P(3, 4);

    buffer->add({
        torch::rand({4, 1, 2, 3}, options1),
        torch::randint(2, {4, 4, 5, 6}, options2)
    }, {x, y, x, y});

    ASSERT_EQ(buffer->size(), 4);

    auto sample = buffer->get({0, 1, 2, 3});
    ASSERT_EQ(sample->objs[0].x, 1); ASSERT_EQ(sample->objs[0].y, 2);
    ASSERT_EQ(sample->objs[1].x, 3); ASSERT_EQ(sample->objs[1].y, 4);
    ASSERT_EQ(sample->objs[2].x, 1); ASSERT_EQ(sample->objs[2].y, 2);
    ASSERT_EQ(sample->objs[3].x, 3); ASSERT_EQ(sample->objs[3].y, 4);

    for (int i = 0; i < 100; i++) {
        buffer->add({
            torch::rand({1, 1, 2, 3}, options1),
            torch::randint(2, {1, 4, 5, 6}, options2)
        }, {x});
    }
    ASSERT_EQ(buffer->size(), 10);

    sample = sampler.sample(100);
    ASSERT_EQ(sample->size(), 100);
    ASSERT_EQ(sample->tensors.size(), 2);
    ASSERT_EQ(sample->tensors[0].size(0), 100);
    ASSERT_EQ(sample->tensors[1].size(0), 100);
}
