#include "rl/logging/client/base.h"

#include <unordered_map>
#include <mutex>


namespace rl::logging::client
{

    static std::mutex mtx{};
    static std::unordered_map<std::string, std::shared_ptr<Base>> loggers{};

    std::shared_ptr<Base> get(const std::string &name)
    {
        std::lock_guard lock{mtx};
        if (loggers.find(name) == loggers.end()) {
            throw std::runtime_error{"Cannot find logger '" + name + "'."};
        }

        return loggers.at(name);
    }

    void create(const std::string &name, std::shared_ptr<Base> logger)
    {
        std::lock_guard lock{mtx};
        if (loggers.find(name) != loggers.end()) {
            throw std::runtime_error{"Logger '" + name + "' already exists."};
        }

        loggers[name] = logger;
    }
}
