#include <torch/torch.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>


torch::Tensor linear_sigmoid(const torch::Tensor &A, const torch::Tensor &x, const torch::Tensor &b)
{
    return torch::sigmoid(A.matmul(x) + b);
}


int main()
{
    torch::StreamGuard stream_guard{at::cuda::getStreamFromPool()};
    auto x = torch::randn({5, 5, 1}).cuda();
    auto A = torch::randn({5, 2, 5}).cuda();
    auto b = torch::randn({5, 2, 1}).cuda();
    auto output1 = linear_sigmoid(A, x, b);
    auto output = output1.square();

    at::cuda::CUDAGraph graph{};
    graph.capture_begin();
    {
        auto intermediate_output = linear_sigmoid(A, x, b);
        output = intermediate_output.square();
    }
    graph.capture_end();
    std::cout << output.mean() << "\n";

    for (int i = 0; i < 10; i++) {
        x.copy_(torch::randn({5, 5, 1}).cuda());
        A.copy_(torch::randn({5, 2, 5}).cuda());
        b.copy_(torch::randn({5, 2, 1}).cuda());
        graph.replay();
        std::cout << output.mean() << "\n";
    }
}
