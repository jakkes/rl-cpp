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

    torch::Tensor compute_gradient_norm(std::shared_ptr<torch::optim::Optimizer> optimizer)
    {
        torch::Tensor grad_norm;
        bool first{true};

        for (const auto &param_group : optimizer->param_groups()) {
            for (const auto &param : param_group.params()) {
                if (first) {
                    grad_norm = param.grad().square().sum();
                    first = false;
                }
                else {
                    grad_norm += param.grad().square().sum();
                }
            }
        }

        return grad_norm.sqrt();
    }
}
