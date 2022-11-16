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
        torch::InferenceMode guard{};
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

    void scale_gradients(std::shared_ptr<torch::optim::Optimizer> optimizer, const torch::Scalar &factor)
    {
        torch::InferenceMode guard{};
        for (auto &param_group : optimizer->param_groups()) {
            for (auto &param : param_group.params()) {
                param.mul_(factor);
            }
        }
    }
}
