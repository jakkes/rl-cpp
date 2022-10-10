#ifndef UTILS_TORCH_H_
#define UTILS_TORCH_H_


#include <memory>

#include <torch/torch.h>

namespace rl::torchutils
{
    bool is_int_dtype(const torch::Tensor &data);

    bool is_bool_dtype(const torch::Tensor &data);

    torch::Tensor compute_gradient_norm(std::shared_ptr<torch::optim::Optimizer> optimizer);

    void scale_gradients(std::shared_ptr<torch::optim::Optimizer> optimizer, const torch::Scalar &factor);
}

#endif /* UTILS_TORCH_H_ */
