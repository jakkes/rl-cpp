#include "rl/env/base.h"


namespace rl::env
{

    void Base::cuda() { is_cuda_ = true; }
    void Base::cpu() { is_cuda_ = false; }

    void Base::set_logger(std::shared_ptr<rl::logging::client::Base> logger) {
        this->logger = logger;
    }

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
