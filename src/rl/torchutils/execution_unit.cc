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
    }

    std::vector<torch::Tensor> ExecutionUnit::operator()(const std::vector<torch::Tensor> &inputs)
    {
        if (!use_cuda_graph) {
            return forward(inputs);
        }

        if (!cuda_graph) {
            init_graph(inputs);
        }

        torch::StreamGuard stream_guard{stream};
        for (int i = 0; i < inputs.size(); i++) {
            this->inputs[i].index_put_({Slice(None, inputs[i].size(0))}, inputs[i]);
        }
        stream_guard.current_stream().synchronize();

        cuda_graph->replay();

        std::vector<torch::Tensor> out{}; out.resize(inputs.size());
        for (int i = 0; i < inputs.size(); i++) {
            out[i] = outputs[i].index({Slice(None, inputs[i].size(0))}).clone();
        }
        stream_guard.current_stream().synchronize();

        return out;
    }
}
