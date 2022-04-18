#include "rl/cpputils/logger.h"

#include <mutex>

#include <spdlog/cfg/env.h>
#include "spdlog/sinks/stdout_color_sinks.h"


namespace rl::cpputils
{
    static std::mutex create_logger_mtx{};
    static bool log_env_loaded{false};

    std::shared_ptr<spdlog::logger> get_logger(const std::string &name)
    {
        std::lock_guard lock{create_logger_mtx};

        if (!log_env_loaded) {
            spdlog::cfg::load_env_levels();
            log_env_loaded = true;
        }

        auto logger = spdlog::get(name);
        if (logger) return logger;

        return spdlog::stdout_color_mt(name);
    }
}
