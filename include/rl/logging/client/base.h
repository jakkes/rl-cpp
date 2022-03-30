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
            virtual void log_scalar(const std::string &name, double value) {
                throw std::runtime_error{"log_scalar not implemented."};
            }

            virtual void log_frequency(const std::string &name, int occurances) {
                throw std::runtime_error{"log_frequency not implemented."};
            }
    };
}

#endif /* RL_LOGGING_HANDLER_BASE_H_ */
