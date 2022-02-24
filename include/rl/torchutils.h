#ifndef UTILS_TORCH_H_
#define UTILS_TORCH_H_


#include <torch/torch.h>

namespace rl::torchutils
{
    bool is_int_dtype(const torch::Tensor &data);

    bool is_bool_dtype(const torch::Tensor &data);
}

#endif /* UTILS_TORCH_H_ */
