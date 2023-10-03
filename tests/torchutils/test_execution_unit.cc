#include <atomic>
#include <thread>

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
            for (int i = 0; i < 1000; i++)
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

    Example x{32, torch::kCUDA};

    torch::Tensor last_output;

    for (int i = 0; i < 100; i++)
    {
        auto input = torch::randn({32, 1000}, torch::TensorOptions{}.device(torch::kCUDA));
        auto output = x({input}).tensors[0];

        if (i > 0) {
            ASSERT_TRUE(
                (output != last_output).any().item().toBool()
            );
        }

        last_output = output;
    }
}


class Rotate2DTransform : public rl::torchutils::ExecutionUnit
{
    public:
        Rotate2DTransform(
            float rotation,
            int max_batchsize,
            torch::Device device
        ) : rl::torchutils::ExecutionUnit{max_batchsize, device}
        {
            A = torch::tensor({
                    {std::cos(rotation), std::sin(rotation)},
                    {-std::sin(rotation), std::cos(rotation)}
                },
                torch::TensorOptions{}.device(device)
            );
        }

    private:
        rl::torchutils::ExecutionUnitOutput forward(const std::vector<torch::Tensor> &inputs) override
        {
            auto &x = inputs[0];

            rl::torchutils::ExecutionUnitOutput output{1, 0};
            output.tensors[0] = torch::matmul(x, A);
            return output;
        }

    private:
        torch::Tensor A;
};


TEST(execution_unit, rotate_90_degrees)
{
    if (!torch::cuda::is_available()) {
        GTEST_SKIP();
    }
    Rotate2DTransform rotation{M_PI_2, 32, torch::kCUDA};
    rotation({torch::randn({2, 2}).cuda()});

    auto x = torch::tensor({{1.0, 0.0}, {0.0, 1.0}}).cuda();
    auto y = rotation({x}).tensors[0];

    ASSERT_TRUE(
        y.allclose(torch::tensor({{0.0, 1.0}, {-1.0, 0.0}}).cuda(), 1e-5, 1e-6)
    );
}


TEST(execution_unit, rotate_90_degrees_two_threads)
{
    std::atomic<bool> running{false};
    Rotate2DTransform rotation{M_PI_2, 32, torch::kCUDA};

    auto worker = [&running, &rotation] () {
        auto x = { torch::tensor({{1.0, 0.0}, {0.0, 1.0}}).cuda() };
        while (!running) {}
        while (running) {
            auto y = rotation(x).tensors[0];

            ASSERT_TRUE(
                y.allclose(torch::tensor({{0.0, 1.0}, {-1.0, 0.0}}).cuda(), 1e-5, 1e-6)
            );
        }
    };

    std::thread thread1{worker};
    std::thread thread2{worker};

    std::this_thread::sleep_for(std::chrono::seconds{1});
    running = true;
    std::this_thread::sleep_for(std::chrono::seconds{1});
    running = false;

    thread1.join();
    thread2.join();
}
