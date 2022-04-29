#ifndef RL_LOGGING_HANDLER_BASE_H_
#define RL_LOGGING_HANDLER_BASE_H_

#include <string>
#include <memory>
#include <stdexcept>


namespace rl::logging::client
{
    /**
     * @brief Base logging client.
     * 
     */
    class Base
    {
        public:
            /**
             * @brief Logs a scalar value.
             * 
             * @param name Log key.
             * @param value Log value.
             */
            virtual void log_scalar(const std::string &name, double value) {
                throw std::runtime_error{"log_scalar not implemented."};
            }

            /**
             * @brief Logs the frequency of which events occur.
             * 
             * @param name Log key.
             * @param occurances Number of occurences to log.
             */
            virtual void log_frequency(const std::string &name, int occurances) {
                throw std::runtime_error{"log_frequency not implemented."};
            }

            /**
             * @brief Logs a text value.
             * 
             * @param name Log key
             * @param value Log value.
             */
            virtual void log_text(const std::string &name, const std::string &value) {
                throw std::runtime_error{"log_text not implemented."};
            }
    };
}

#endif /* RL_LOGGING_HANDLER_BASE_H_ */
