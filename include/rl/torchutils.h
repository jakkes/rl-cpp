#ifndef UTILS_TORCH_H_
#define UTILS_TORCH_H_


#include <memory>

#include <torch/torch.h>

namespace rl::torchutils
{
    bool is_int_dtype(const torch::Tensor &data);

    bool is_bool_dtype(const torch::Tensor &data);

    torch::Tensor compute_gradient_norm(std::shared_ptr<torch::optim::Optimizer> optimizer);
}

#endif /* UTILS_TORCH_H_ */
