#include <gtest/gtest.h>
#include <rl/rl.h>

#include "torch_test.h"


using namespace rl::policies;

TORCH_TEST(policies, sample_gamma, device)
{
    auto o = torch::TensorOptions{}.device(device).dtype(torch::kFloat64);

    Gamma g{
        torch::tensor({0.5, 1.0, 2.0}, o),
        torch::ones({3}, o)
    };

    double a{0}, b{0}, c{0};

    static int N{100000};
    for (int i = 0; i < N; i++) {
        auto sample = g.sample();
        a += 1.0 / N * sample.index({0}).item().toDouble();
        b += 1.0 / N * sample.index({1}).item().toDouble();
        c += 1.0 / N * sample.index({2}).item().toDouble();
    }

    std::cout << a << "\n";
    std::cout << b << "\n";
    std::cout << c << "\n";
}
