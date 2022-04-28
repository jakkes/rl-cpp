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
        
        re->set_logger(logger);

        if (is_cuda) re->cuda();
        else re->cpu();

        return re;
    }

    void Factory::set_logger(std::shared_ptr<rl::logging::client::Base> logger) {
        this->logger = logger;
    }

    void Factory::cuda() { is_cuda = true; }
    void Factory::cpu() { is_cuda = false; }
}
