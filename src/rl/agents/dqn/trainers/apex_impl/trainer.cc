#include "trainer.h"

#include <c10/cuda/CUDAStream.h>

#include <rl/cpputils/logger.h>
#include <rl/torchutils/torchutils.h>


namespace rl::agents::dqn::trainers::apex_impl
{
    static
    auto LOGGER = rl::cpputils::get_logger("ApexDQN-Trainer");

    Trainer::Trainer(
        std::shared_ptr<rl::agents::dqn::Module> module,
        std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        std::shared_ptr<rl::buffers::Tensor> replay_buffer,
        const ApexOptions &options
    ) : options{options}
    {
        this->module = module;
        this->target_module = std::dynamic_pointer_cast<rl::agents::dqn::Module>(module->clone());
        this->value_parser = value_parser;
        this->optimizer = optimizer;
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
            target_network_update();
        }
        LOGGER->info("Training stopped");
    }

    void Trainer::step()
    {
        auto sample_storage = replay_buffer->sample(options.batch_size);
        auto &samples = *sample_storage;

        auto outputs = module->forward(samples[0].to(options.network_device));
        auto masks = samples[1].to(options.network_device);

        torch::Tensor next_actions;
        torch::Tensor next_outputs;
        auto next_masks = samples[6].to(options.network_device);
        auto next_states = samples[5].to(options.network_device);
        {
            torch::InferenceMode guard{};
            next_outputs = target_module->forward(next_states);

            if (options.double_dqn) {
                auto tmp_output = module->forward(next_states);
                next_actions = value_parser->values(tmp_output, next_masks).argmax(-1);
            } else {
                next_actions = value_parser->values(next_outputs, next_masks).argmax(-1);
            }
        }

        auto loss = value_parser->loss(
            outputs,
            masks,
            samples[2].to(options.network_device),
            samples[3].to(options.network_device),
            samples[4].to(options.network_device),
            next_outputs,
            next_masks,
            next_actions,
            std::pow(options.discount, options.n_step)
        );

        loss = loss.mean();
        optimizer->zero_grad();
        loss.backward();
        auto grad_norm = rl::torchutils::compute_gradient_norm(optimizer);
        auto grad_norm_factor = torch::where(
            grad_norm > options.max_gradient_norm,
            options.max_gradient_norm / grad_norm,
            torch::ones_like(grad_norm)
        );
        rl::torchutils::scale_gradients(optimizer, grad_norm_factor);
        optimizer->step();

        if (options.logger) {
            options.logger->log_scalar("ApexDQN/Loss", loss.item().toFloat());
            options.logger->log_scalar("ApexDQN/Gradient norm", grad_norm.item().toFloat());
            options.logger->log_frequency("ApexDQN/Update frequency", 1);
        }
    }

    void Trainer::target_network_update()
    {
        torch::InferenceMode guard{};
        auto target_parameters = target_module->parameters();
        auto parameters = module->parameters();

        for (int i = 0; i < parameters.size(); i++) {
            target_parameters[i].add_(parameters[i] - target_parameters[i], options.target_network_lr);
        }
    }
}
