#ifndef RL_LOGGING_HANDLER_H_
#define RL_LOGGING_HANDLER_H_


#include <string>
#include <memory>
#include <unordered_map>

#include "client/base.h"


namespace rl::logging
{
    class Handler
    {
        public:
            static void create(std::string_view name, std::shared_ptr<client::Base> logger);
            static std::shared_ptr<client::Base> get(std::string_view);

        private:
            static std::unordered_map<std::string, std::shared_ptr<client::Base> loggers;
    };
}

#endif /* RL_LOGGING_HANDLER_H_ */
