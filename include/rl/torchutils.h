#ifndef UTILS_TORCH_H_
#define UTILS_TORCH_H_


#include <torch/torch.h>

namespace rl::torchutils
{
    bool is_int_dtype(const torch::Tensor &data);

    bool is_bool_dtype(const torch::Tensor &data);

    torch::IntArrayRef slice_shape(const torch::IntArrayRef &shape, int x);
}

#endif /* UTILS_TORCH_H_ */
