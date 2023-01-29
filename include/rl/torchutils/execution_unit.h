#ifndef RL_TORCHUTILS_CUDA_GRAPH_UNIT_H_
#define RL_TORCHUTILS_CUDA_GRAPH_UNIT_H_


#include <vector>

#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGraph.h>
#include <torch/torch.h>

namespace rl::torchutils
{
    class ExecutionUnit
    {
        public:
            ExecutionUnit(bool use_cuda_graph, int max_batchsize);

            std::vector<torch::Tensor> operator()(const std::vector<torch::Tensor> &inputs);

        private:
            const bool use_cuda_graph;
            const int max_batchsize;
            c10::cuda::CUDAStream stream{c10::cuda::getStreamFromPool()};
            std::unique_ptr<at::cuda::CUDAGraph> cuda_graph = nullptr;

            std::vector<torch::Tensor> inputs;
            std::vector<torch::Tensor> outputs;

        private:
            virtual
            std::vector<torch::Tensor> forward(const std::vector<torch::Tensor> &inputs) = 0;

            void init_graph(const std::vector<torch::Tensor> &inputs);
    };
}

#endif /* RL_TORCHUTILS_CUDA_GRAPH_UNIT_H_ */
