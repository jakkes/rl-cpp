#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGraph.h>
#include <torch/torch.h>


int main()
{
    torch::StreamGuard stream_guard{at::cuda::getStreamFromPool()};
    at::cuda::CUDAGraph graph{};
    auto options = torch::TensorOptions{}.device(torch::kCUDA);

    auto x = torch::randn({25, 5}, options);
    c10::List<c10::optional<at::Tensor>> index_list{x < 0};
    x = x.index_put(index_list, - x.index(index_list));
    auto y = x.square();

    x.copy_(torch::randn({25, 5}, options));
    graph.capture_begin();
    c10::List<c10::optional<at::Tensor>> index_list{x < 0};
    x = x.index_put(index_list, - x.index(index_list));
    y = x.square();
    graph.capture_end();

    std::cout << y;

    x.copy_(torch::randn({25, 5}, options));
    graph.replay();
    std::cout << y;
    x.copy_(torch::randn({25, 5}, options));
    graph.replay();
    std::cout << y;
    x.copy_(torch::randn({25, 5}, options));
    graph.replay();
    std::cout << y;
}
