#include "rl/buffers/tensor.h"

#include <stdexcept>


namespace rl::buffers
{

    Tensor::Tensor(
        int64_t capacity,
        const std::vector<std::vector<int64_t>> &tensor_shapes,
        const std::vector<torch::TensorOptions> &tensor_options
    ) : capacity{capacity}
    {
        if (tensor_shapes.size() != tensor_options.size()) {
            throw std::invalid_argument{"Tensor shapes and options must be of same length."};
        }
        auto n = tensor_shapes.size();
        data.reserve(n);

        for (int i = 0; i < n; i++) {
            std::vector<int64_t> shape;
            shape.reserve(tensor_shapes[i].size() + 1);

            shape.push_back(capacity);
            shape.insert(shape.end(), tensor_shapes[i].begin(), tensor_shapes[i].end());

            data.push_back(torch::zeros(shape, tensor_options[i]));
        }
    }

    int64_t Tensor::size() const {
        if (looped) return capacity;
        return memory_index;
    }

    std::unique_ptr<std::vector<torch::Tensor>> Tensor::get(torch::Tensor indices)
    {
        auto re = std::make_unique<std::vector<torch::Tensor>>();
        re->reserve(data.size());

        std::lock_guard<std::mutex> guard{lock};

        for (int i = 0; i < data.size(); i++) {
            re->push_back(data[i].index({indices}));
        }

        return re;
    }

    std::unique_ptr<std::vector<torch::Tensor>> Tensor::get(const std::vector<int64_t> &indices) {
        return get(torch::tensor(indices, torch::TensorOptions{}.dtype(torch::kLong)));
    }

    void Tensor::add(const std::vector<torch::Tensor> &data)
    {
        if (data.size() != this->data.size()) {
            throw std::invalid_argument{"Invalid number of tensors."};
        }

        auto bs = data[0].size(0);
        for (const auto &t : data) assert(t.size(0) == bs);

        std::lock_guard<std::mutex> guard{lock};

        auto indices = (torch::arange(bs) + static_cast<int64_t>(memory_index)) % capacity;
        
        for (int i = 0; i < data.size(); i++) {
            this->data[i].index_put_(
                {indices},
                data[i]
            );
        }

        memory_index += bs;
        if (memory_index > capacity) {
            memory_index = memory_index % capacity;
            looped = true;
        }
    }
}
