#ifndef RL_TORCHUTILS_CUDA_GRAPH_UNIT_H_
#define RL_TORCHUTILS_CUDA_GRAPH_UNIT_H_


#include <vector>

#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGraph.h>
#include <torch/torch.h>

namespace rl::torchutils
{
    struct ExecutionUnitOutput
    {
        std::vector<torch::Tensor> tensors;
        std::vector<torch::Tensor> scalars;

        ExecutionUnitOutput() = default;
        ExecutionUnitOutput(size_t tensors, size_t scalars);

        ExecutionUnitOutput clone(int batchsize);
    };

    class ExecutionUnit
    {
        public:
            ExecutionUnit(bool use_cuda_graph, int max_batchsize);

            ExecutionUnitOutput operator()(const std::vector<torch::Tensor> &inputs);

        private:
            const bool use_cuda_graph;
            const int max_batchsize;
            c10::cuda::CUDAStream stream{c10::cuda::getStreamFromPool()};
            std::unique_ptr<at::cuda::CUDAGraph> cuda_graph = nullptr;
     
            std::vector<torch::Tensor> inputs;
            ExecutionUnitOutput outputs;
    
        private:
            virtual
            ExecutionUnitOutput forward(const std::vector<torch::Tensor> &inputs) = 0;

            void init_graph(const std::vector<torch::Tensor> &inputs);
    };
}

#endif /* RL_TORCHUTILS_CUDA_GRAPH_UNIT_H_ */
