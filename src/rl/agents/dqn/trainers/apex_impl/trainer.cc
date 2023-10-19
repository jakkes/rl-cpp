#include "trainer.h"

#include <c10/cuda/CUDAStream.h>

#include <rl/cpputils/logger.h>
#include <rl/torchutils/torchutils.h>


namespace rl::agents::dqn::trainers::apex_impl
{
    static
    auto LOGGER = rl::cpputils::get_logger("ApexDQN-Trainer");

    Trainer::Trainer(
        std::shared_ptr<TrainingUnit> training_unit,
        std::shared_ptr<rl::buffers::Tensor> replay_buffer,
        const ApexOptions &options
    ) : options{options}
    {
        this->training_unit = training_unit;
        this->replay_buffer = std::make_shared<rl::buffers::samplers::Uniform<rl::buffers::Tensor>>(replay_buffer);
    }

    void Trainer::start()
    {
        running = true;
        working_thread = std::thread(&Trainer::worker, this);
    }

    void Trainer::stop()
    {
        running = false;
        if (working_thread.joinable()) working_thread.join();
    }

    void Trainer::worker()
    {
        torch::StreamGuard stream_guard{c10::cuda::getStreamFromPool()};

        while (running && replay_buffer->buffer_size() < options.minimum_replay_buffer_size) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        LOGGER->info("Starting training");
        while (running) {
            step();
        }
        LOGGER->info("Training stopped");
    }

    void Trainer::step()
    {
        auto sample_storage = replay_buffer->sample(options.batch_size);
        auto &samples = *sample_storage;

        auto metrics = training_unit->operator()({
            samples[0].to(options.network_device),
            samples[1].to(options.network_device),
            samples[2].to(options.network_device),
            samples[3].to(options.network_device),
            samples[4].to(options.network_device),
            samples[5].to(options.network_device),
            samples[6].to(options.network_device),
        });

        if (options.logger) {
            options.logger->log_scalar("ApexDQN/Loss", metrics.scalars[0].item().toFloat());
            options.logger->log_scalar("ApexDQN/Gradient norm", metrics.scalars[1].item().toFloat());
            options.logger->log_frequency("ApexDQN/Update frequency", 1);
        }
    }
}
