#ifndef RL_LOGGING_CLIENT_TENSORBOARD_H_
#define RL_LOGGING_CLIENT_TENSORBOARD_H_


#include <mutex>
#include <unordered_map>
#include <string>

#include <tensorboard_logger.h>

#include "base.h"
#include "rl/option.h"
#include "rl/cpputils/metronome.h"


namespace rl::logging::client
{
    struct TensorboardOptions
    {
        RL_OPTION(size_t, frequency_window) = 5;
    };

    class Tensorboard : public Base
    {
        public:
            Tensorboard(const TensorboardOptions &options={});
            ~Tensorboard() = default;

            void log_scalar(const std::string &name, double value) override;
            void log_frequency(const std::string &name, int occurances) override;
            void log_text(const std::string &name, const std::string &value) override;

        private:
            const TensorboardOptions options;

            TensorBoardLogger logger;
            std::recursive_mutex mtx{};
            std::unordered_map<std::string, size_t> steps{};
            std::unordered_map<std::string, size_t> occurances{};
            
            rl::cpputils::Metronome<std::chrono::seconds> metronome;

            void metronome_callback();
    };
}

#endif /* RL_LOGGING_CLIENT_TENSORBOARD_H_ */
