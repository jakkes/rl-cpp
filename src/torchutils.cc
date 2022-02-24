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
}
