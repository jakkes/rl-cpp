#ifndef RL_TORCHUTILS_SCALE_GRADIENTS_H_
#define RL_TORCHUTILS_SCALE_GRADIENTS_H_


#include <torch/torch.h>


namespace rl::torchutils
{
    inline
    void scale_gradients(std::shared_ptr<torch::optim::Optimizer> optimizer, const torch::Tensor &factor)
    {
        torch::NoGradGuard guard{};
        for (auto &param_group : optimizer->param_groups()) {
            for (auto &param : param_group.params()) {
                param.mul_(factor);
            }
        }
    }
}

#endif /* RL_TORCHUTILS_SCALE_GRADIENTS_H_ */
