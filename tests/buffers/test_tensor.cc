#include <torch/torch.h>
#include <gtest/gtest.h>

#include "rl/rl.h"

using namespace rl;


void run_tensor(torch::Device device)
{
    std::vector<int64_t> shape1 = {1, 2, 3};
    auto options1 = torch::TensorOptions{}.dtype(torch::kFloat32).device(device);

    std::vector<int64_t> shape2 = {4, 5, 6};
    auto options2 = torch::TensorOptions{}.dtype(torch::kBool).device(device);

    auto shapes = {shape1, shape2};
    auto options = {options1, options2};
    
    auto buffer = std::make_shared<buffers::Tensor>(10, shapes, options);
    auto sampler = buffers::samplers::Uniform<buffers::Tensor>(buffer);

    ASSERT_EQ(buffer->size(), 0);

    buffer->add({
        torch::rand({4, 1, 2, 3}, options1),
        torch::randint(2, {4, 4, 5, 6}, options2)
    });

    ASSERT_EQ(buffer->size(), 4);

    for (int i = 0; i < 100; i++) {
        buffer->add({
            torch::rand({1, 1, 2, 3}, options1),
            torch::randint(2, {1, 4, 5, 6}, options2)
        });
    }
    ASSERT_EQ(buffer->size(), 10);
    buffer->clear();
    ASSERT_EQ(buffer->size(), 0);
    for (int i = 0; i < 20; i++) {
        buffer->add({
            torch::rand({1, 1, 2, 3}, options1),
            torch::randint(2, {1, 4, 5, 6}, options2)
        });
    }

    auto sample = sampler.sample(100);
    ASSERT_EQ(sample->size(), 2);
    ASSERT_EQ((*sample)[0].size(0), 100);
    ASSERT_EQ((*sample)[1].size(0), 100);
}

TEST(test_buffers, test_tensor_cpu)
{
    run_tensor(torch::kCPU);
}

TEST(test_buffers, test_tensor_cuda)
{
    if (!torch::cuda::is_available()) GTEST_SKIP();
    run_tensor(torch::kCUDA);
}
