#include "rl/env/base.h"


namespace rl::env
{

    std::unique_ptr<Base> Factory::get() const
    {
        auto re = get_impl();
        if (is_cuda) {
            re->cuda();
            assert(re->state()->state.device().type() == torch::kCUDA);
        }
        else {
            re->cpu();
            assert(re->state()->state.device().type() == torch::kCPU);
        }

        return re;
    }

    void Factory::cuda() { is_cuda = true; }
    void Factory::cpu() { is_cuda = false; }
}
