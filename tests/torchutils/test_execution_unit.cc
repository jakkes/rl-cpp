#include <torch/torch.h>
#include <gtest/gtest.h>
#include <rl/torchutils/execution_unit.h>



class Example : public rl::torchutils::ExecutionUnit
{
    public:
        using rl::torchutils::ExecutionUnit::ExecutionUnit;

    private:
        std::vector<torch::Tensor> forward(const std::vector<torch::Tensor> &inputs)
        {
            auto x = inputs[0];
            for (int i = 0; i < 10000; i++)
            {
                x = (x - x.mean(1, true)) / x.std(1, true, true);
                x = x.sigmoid();
            }

            return {x};
        }
};


TEST(execution_unit, simple)
{
    if (!torch::cuda::is_available()) {
        GTEST_SKIP();
    }

    Example x{true, 32};

    std::vector<size_t> times{};

    for (int i = 0; i < 10; i++)
    {
        auto input = torch::randn({32, 10000}, torch::TensorOptions{}.device(torch::kCUDA));
        auto tic = std::chrono::high_resolution_clock::now();
        auto output = x({input})[0];
        auto toc = std::chrono::high_resolution_clock::now();

        times.push_back((toc-tic).count());
    }

    for (auto ns : times) {
        std::cout << ns << "\n";
    }
}
