#include "rl/logging/client/tensorboard.h"

#include <filesystem>


namespace rl::logging::client
{
    static
    std::string get_name() {
        if (!std::filesystem::exists("runs")) {
            std::filesystem::create_directory("runs");
        }

        size_t count{0};
        for (const auto &entry : std::filesystem::directory_iterator("runs")) {
            count++;
        }
        std::filesystem::create_directory("runs/run_" + std::to_string(count));

        return "runs/run_" + std::to_string(count) + "/logs.tfevents";
    }

    Tensorboard::Tensorboard(const TensorboardOptions &options)
    : logger{get_name()}, options{options}, metronome{options.frequency_window}
    {
        metronome.start(std::bind(&Tensorboard::metronome_callback, this));
    }

    void Tensorboard::log_scalar(const std::string &name, double value) {
        std::lock_guard lock{mtx};
        logger.add_scalar(name, steps[name]++, value);
    }

    void Tensorboard::log_text(const std::string &name, const std::string &value) {
        std::lock_guard lock{mtx};
        logger.add_text(name, steps[name]++, value.c_str());
    }

    void Tensorboard::log_frequency(const std::string &name, int occurances) {
        std::lock_guard lock{mtx};
        this->occurances[name] += occurances;
    }

    void Tensorboard::metronome_callback() {
        std::lock_guard lock{mtx};
        for (const auto &item : occurances) {
            log_scalar(item.first, item.second * 1.0 / options.frequency_window);
            occurances[item.first] = 0;
        }
    }
}
