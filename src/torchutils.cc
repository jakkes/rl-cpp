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

    /**
     * @brief Slices a shape array.
     * 
     * @param shape Shape to array
     * @param x slice value. If non negative, this is equal returns `shape[x:]`. If it is
     *  negative, `shape[:x]` is returned.
     * @return torch::IntArrayRef 
     */
    torch::IntArrayRef slice_shape(const torch::IntArrayRef &shape, int n)
    {
        int offset = n > 0 ? n : 0;
        n = n > 0 ? n : -n;
        auto size = shape.size() - n;
        std::vector<int64_t> sliced{};
        sliced.reserve(size);
        for (int i = 0; i < size; i++) {
            sliced.push_back(shape[i + offset]);
        }
        return torch::IntArrayRef{sliced};
    }
}
