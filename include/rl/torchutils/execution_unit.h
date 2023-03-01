#ifndef RL_TORCHUTILS_CUDA_GRAPH_UNIT_H_
#define RL_TORCHUTILS_CUDA_GRAPH_UNIT_H_


#include <vector>
#include <mutex>

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
            ExecutionUnit(bool use_cuda_graph, int max_batchsize, torch::Device device=torch::kCPU);

            ExecutionUnitOutput operator()(const std::vector<torch::Tensor> &inputs);

        protected:
            const int batchsize;
            torch::Device device;

        private:
            bool use_cuda_graph;
            std::mutex mtx{};

            std::unique_ptr<c10::cuda::CUDAStream> stream;
            std::unique_ptr<at::cuda::CUDAGraph> cuda_graph = nullptr;
     
            std::vector<torch::Tensor> inputs;
            ExecutionUnitOutput outputs;
    
        private:
            virtual
            ExecutionUnitOutput forward(const std::vector<torch::Tensor> &inputs) = 0;

            void init_graph(const std::vector<torch::Tensor> &inputs);
    };

    class ExecutionUnitLoadBalancer
    {
        public:
            ExecutionUnitLoadBalancer(
                const std::vector<std::shared_ptr<ExecutionUnit>> &execution_units
            );

            inline
            ExecutionUnitOutput operator()(const std::vector<torch::Tensor> &inputs) {
                return get_next_execution_unit()(inputs);
            }

            inline 
            void run_all(const std::vector<torch::Tensor> &inputs) {
                for (auto &unit : execution_units) {
                    unit->operator()(inputs);
                }
            }

        private:
            std::vector<std::shared_ptr<ExecutionUnit>> execution_units;
            size_t current_index{0};
            std::mutex mtx{};

        private:
            inline
            ExecutionUnit &get_next_execution_unit() {
                std::lock_guard lock{mtx};
                auto &out = *execution_units[current_index++];
                if (current_index >= execution_units.size()) {
                    current_index = 0;
                }
                return out;
            }
    };
}

#endif /* RL_TORCHUTILS_CUDA_GRAPH_UNIT_H_ */
