#ifndef RL_LOGGING_HANDLER_BASE_H_
#define RL_LOGGING_HANDLER_BASE_H_

#include <string>
#include <memory>
#include <stdexcept>


namespace rl::logging::client
{
    class Base
    {
        public:
            virtual void log_scalar(const std::string &name, float value) {
                throw std::runtime_error{"log_scalar not implemented."};
            }
    };

    std::shared_ptr<Base> get(const std::string &name);
    void create(const std::string &name, std::shared_ptr<Base> logger);
}

#endif /* RL_LOGGING_HANDLER_BASE_H_ */
