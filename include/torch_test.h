#include <torch/torch.h>
#include <gtest/gtest.h>

#ifndef TORCH_TEST_H_
#define TORCH_TEST_H_


#define TORCH_TEST(test_suite_name, test_name, device) \
void run_##test_suite_name##test_name(torch::Device device); \
TEST(test_suite_name, test_name##_cpu) { run_##test_suite_name##test_name(torch::kCPU); } \
TEST(test_suite_name, test_name##_gpu) { if (!torch::cuda::is_available()) GTEST_SKIP(); run_##test_suite_name##test_name(torch::kCUDA); } \
void run_##test_suite_name##test_name(torch::Device device)

#endif /* TORCH_TEST_H_ */
