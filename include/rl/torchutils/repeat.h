#ifndef RL_TORCHUTILS_REPEAT_H_
#define RL_TORCHUTILS_REPEAT_H_


#include <torch/torch.h>


namespace rl::torchutils
{
    inline
    torch::Tensor repeat(const torch::Tensor &x, const std::vector<int64_t> &repeats)
    {
        auto N = x.sizes().size();
        std::vector<int64_t> repeat_vector{};
        repeat_vector.resize(N);

        int i = 0;
        for (auto &repeat : repeats) {
            repeat_vector[i++] = repeat;
        }
        while (i < N) {
            repeat_vector[i++] = 1;
        }

        return x.repeat(repeat_vector);
    }
}

#endif /* RL_TORCHUTILS_REPEAT_H_ */
