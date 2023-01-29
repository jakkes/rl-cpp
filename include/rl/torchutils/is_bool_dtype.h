#ifndef RL_TORCHUTILS_IS_BOOL_DTYPE_H_
#define RL_TORCHUTILS_IS_BOOL_DTYPE_H_


#include <torch/torch.h>

namespace rl::torchutils
{
    inline
    bool is_bool_dtype(const torch::Tensor &data) {
        return data.dtype() == torch::kBool;
    }
}

#endif /* RL_TORCHUTILS_IS_BOOL_DTYPE_H_ */
