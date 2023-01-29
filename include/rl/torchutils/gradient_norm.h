#ifndef RL_TORCHUTILS_GRADIENT_NORM_H_
#define RL_TORCHUTILS_GRADIENT_NORM_H_

#include <memory>

#include <torch/torch.h>


namespace rl::torchutils
{
    inline
    torch::Tensor compute_gradient_norm(std::shared_ptr<torch::optim::Optimizer> optimizer)
    {
        torch::NoGradGuard guard{};
        torch::Tensor grad_norm;
        bool first{true};

        for (const auto &param_group : optimizer->param_groups()) {
            for (const auto &param : param_group.params()) {
                auto &grad = param.grad();
                if (!grad.defined()) {
                    continue;
                }

                if (first) {
                    grad_norm = grad.square().sum();
                    first = false;
                }
                else {
                    grad_norm += grad.square().sum();
                }
            }
        }

        return grad_norm.sqrt();
    }
}

#endif /* RL_TORCHUTILS_GRADIENT_NORM_H_ */
