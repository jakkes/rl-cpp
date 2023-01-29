#include <torch/torch.h>
#include <gtest/gtest.h>
#include <rl/torchutils/execution_unit.h>


class Example : public rl::torchutils::ExecutionUnit
{
    public:
        using rl::torchutils::ExecutionUnit::ExecutionUnit;

    private:
        rl::torchutils::ExecutionUnitOutput forward(const std::vector<torch::Tensor> &inputs)
        {
            auto x = inputs[0];
            for (int i = 0; i < 10000; i++)
            {
                x = (x - x.mean(1, true)) / x.std(1, true, true);
                x = x.sigmoid();
            }

            rl::torchutils::ExecutionUnitOutput out{1, 0};
            out.tensors[0] = x;
            return out;
        }
};


TEST(execution_unit, simple)
{
    if (!torch::cuda::is_available()) {
        GTEST_SKIP();
    }

    Example x{true, 32};

    torch::Tensor last_output;

    for (int i = 0; i < 10; i++)
    {
        auto input = torch::randn({32, 10000}, torch::TensorOptions{}.device(torch::kCUDA));
        auto output = x({input}).tensors[0];

        if (i > 0) {
            ASSERT_TRUE(
                (output != last_output).any().item().toBool()
            );
        }

        last_output = output;
    }
}
