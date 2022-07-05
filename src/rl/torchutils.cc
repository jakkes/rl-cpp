#include "rl/torchutils.h"


namespace rl::torchutils
{
    static std::set<torch::ScalarType> allowed_dtypes{{ torch::kInt64, torch::kInt32, torch::kInt16, torch::kInt8 }};
    bool is_int_dtype(const torch::Tensor &data)
    {
        return allowed_dtypes.find(data.dtype().toScalarType()) != allowed_dtypes.end();
    }

    bool is_bool_dtype(const torch::Tensor &data)
    {
        return data.dtype() == torch::kBool;
    }

    torch::Tensor &TensorHolder::register_tensor(const std::string &name, const torch::Tensor &tensor)
    {
        if (tensors.find(name) != tensors.end()) {
            throw std::invalid_argument{"A tensor was already registered with the given name."};
        }

        tensors.insert({name, tensor});
        return tensor;
    }

    void TensorHolder::to(torch::Device device)
    {
        for (auto tensor : tensors) {
            tensor.second.to(device);
        }
    }
}
