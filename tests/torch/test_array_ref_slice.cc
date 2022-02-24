#include <iostream>

#include <gtest/gtest.h>
#include <torch/torch.h>


TEST(test_torch, test_array_ref_slice)
{
    auto x1 = torch::rand({1,2,3,4}).sizes().slice(1);
    auto x2 = torch::rand({1,2,3,4}).sizes().slice(2);
    auto x3 = torch::rand({1,2,3,4}).sizes().slice(3);
    auto x4 = torch::rand({1,2,3,4}).sizes().slice(4);
    auto xm1 = torch::rand({1,2,3,4}).sizes().slice(-1);

    for (auto v : x1) std::cout << v;
    std::cout << "\n";

    for (auto v : x2) std::cout << v;
    std::cout << "\n";

    for (auto v : x3) std::cout << v;
    std::cout << "\n";

    for (auto v : x4) std::cout << v;
    std::cout << "\n";

    for (auto v : xm1) std::cout << v;
    std::cout << "\n";

}
