#include "rl/torchutils/execution_unit.h"

#include <unordered_set>


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
        std::vector<int64_t> repeats{};
        repeats.push_back(batchsize - n);
        for (int i = 0; i < input.sizes().size() - 1; i++) {
            repeats.push_back(1);
        }
        
        return torch::concat(
            {
                input,
                input.index({0}).unsqueeze(0).repeat(repeats)
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

    ExecutionUnit::ExecutionUnit(int max_batchsize, c10::Device device, bool enable_cuda_graph, bool use_high_priority_stream)
    : 
        enable_cuda_graph{enable_cuda_graph},
        max_batchsize{max_batchsize},
        device{device}
    {
        if (device.is_cuda() && enable_cuda_graph) {
            stream = std::make_unique<c10::cuda::CUDAStream>(
                c10::cuda::getStreamFromPool(use_high_priority_stream, device.index())
            );
        }
    }

    void ExecutionUnit::init_graph(const std::vector<torch::Tensor> &inputs)
    {
        torch::StreamGuard stream_guard{*stream};
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
        if (!stream) {
            return forward(inputs);
        }

        std::lock_guard lock{mtx};

        // Synchronize input streams as we are about to enter another one.
        c10::cuda::getCurrentCUDAStream(device.index()).synchronize();
        auto batchsize = inputs.size() == 0 ? max_batchsize : inputs[0].size(0);
        if (batchsize > max_batchsize) {
            throw std::invalid_argument{
                "Cannot execute a batch larger than the given batchsize. Received "
                "batch of size " + std::to_string(batchsize) + ", configured max "
                "batchsize is " + std::to_string(this->max_batchsize) + "."
            };
        }

        if (!cuda_graph) {
            init_graph(inputs);
        }

        torch::StreamGuard stream_guard{*stream};
        for (int i = 0; i < inputs.size(); i++) {
            this->inputs[i].index_put_({Slice(None, batchsize)}, inputs[i]);
        }

        // TODO: Do we need to synchronize here? Probably not.
        stream_guard.current_stream().synchronize();
        cuda_graph->replay();
        auto out = outputs.clone(batchsize);
        stream_guard.current_stream().synchronize();

        return out;
    }

    ExecutionUnitLoadBalancer::ExecutionUnitLoadBalancer(
        const std::vector<std::shared_ptr<ExecutionUnit>> &execution_units
    ) : execution_units{execution_units}
    {}
}
