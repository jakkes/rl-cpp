#ifndef RL_LOGGING_HANDLER_BASE_H_
#define RL_LOGGING_HANDLER_BASE_H_

#include <string>


namespace rl::logging::client
{
    class Base
    {
        public:
            virtual void log_scalar(std::string_view name, float value) = 0;
    };
}

#endif /* RL_LOGGING_HANDLER_BASE_H_ */
