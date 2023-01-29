#ifndef RL_TORCHUTILS_IS_INT_DTYPE_H_
#define RL_TORCHUTILS_IS_INT_DTYPE_H_


#include <torch/torch.h>

namespace rl::torchutils
{
    inline
    bool is_int_dtype(const torch::Tensor &data) {
        auto dtype = data.dtype().toScalarType();
        return dtype == torch::kInt64 || dtype == torch::kInt32 || dtype == torch::kInt16 || dtype == torch::kInt8;
    }
}

#endif /* RL_TORCHUTILS_IS_INT_DTYPE_H_ */
