#include "rl/torchutils/execution_unit.h"


using namespace torch::indexing;

namespace rl::torchutils
{
    static
    torch::Tensor expand_to_batchsize(const torch::Tensor &input, int batchsize)
    {
        if (input.size(0) == batchsize) {
            return input;
        }

        auto n = input.size(0);
        
        return torch::concat(
            {
                input,
                input.index({0}).unsqueeze(0).repeat({batchsize - n})
            },
            0
        );
    }

    ExecutionUnitOutput::ExecutionUnitOutput(size_t tensors, size_t scalars) {
        this->tensors.resize(tensors);
        this->scalars.resize(scalars);
    }

    ExecutionUnitOutput ExecutionUnitOutput::clone(int batchsize) {
        ExecutionUnitOutput out{tensors.size(), scalars.size()};
        for (int i = 0; i < tensors.size(); i++) {
            out.tensors[i] = tensors[i].index({Slice(None, batchsize)}).clone();
        }
        for (int i = 0; i < scalars.size(); i++) {
            out.scalars[i] = scalars[i].clone();
        }
        return out;
    }

    ExecutionUnit::ExecutionUnit(bool use_graph, int max_batchsize)
    : use_cuda_graph{use_graph}, max_batchsize{max_batchsize}
    {}

    void ExecutionUnit::init_graph(const std::vector<torch::Tensor> &inputs)
    {
        torch::StreamGuard stream_guard{stream};
        cuda_graph = std::make_unique<at::cuda::CUDAGraph>();

        this->inputs.reserve(inputs.size());
        for (auto &input : inputs) {
            this->inputs.push_back(expand_to_batchsize(input, max_batchsize));
        }

        outputs = forward(this->inputs);
        stream_guard.current_stream().synchronize();

        cuda_graph->capture_begin();
        outputs = forward(this->inputs);
        cuda_graph->capture_end();
        stream_guard.current_stream().synchronize();
    }

    ExecutionUnitOutput ExecutionUnit::operator()(const std::vector<torch::Tensor> &inputs)
    {
        if (!use_cuda_graph) {
            return forward(inputs);
        }

        std::lock_guard lock{mtx};

        // Synchronize current stream as we are about to enter another one.
        c10::cuda::getCurrentCUDAStream().synchronize();
        auto batchsize = inputs.size() == 0 ? max_batchsize : inputs[0].size(0);
        if (batchsize > max_batchsize) {
            throw std::invalid_argument{
                "Cannot execute a batch larger than the given batchsize. Received "
                "batch of size " + std::to_string(batchsize) + ", configured max "
                "batchsize is " + std::to_string(max_batchsize) + "."
            };
        }

        if (!cuda_graph) {
            init_graph(inputs);
        }

        torch::StreamGuard stream_guard{stream};
        for (int i = 0; i < inputs.size(); i++) {
            this->inputs[i].index_put_({Slice(None, batchsize)}, inputs[i]);
        }

        // TODO: Do we need to synchronize here?
        stream_guard.current_stream().synchronize();

        cuda_graph->replay();

        auto out = outputs.clone(batchsize);
        stream_guard.current_stream().synchronize();

        return out;
    }
}
