#ifndef RL_CPPUTILS_LOGGER_H_
#define RL_CPPUTILS_LOGGER_H_


#include <spdlog/spdlog.h>

namespace rl::cpputils
{
    std::shared_ptr<spdlog::logger> get_logger(const std::string &name);
}

#endif /* RL_CPPUTILS_LOGGER_H_ */
