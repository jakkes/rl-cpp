#include "rl/env/base.h"

#include <algorithm>
#include <sstream>

namespace rl::env
{
    void Base::set_logger(std::shared_ptr<rl::logging::client::Base> logger) {
        this->logger = logger;
    }

    std::unique_ptr<Base> Factory::get() const
    {
        auto re = get_impl();
        re->set_logger(logger);
        return re;
    }

    void Factory::set_logger(std::shared_ptr<rl::logging::client::Base> logger) {
        this->logger = logger;
    }
}
