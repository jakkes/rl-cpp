#include "trainer.h"

#include <c10/cuda/CUDAStream.h>
#include <rl/torchutils/torchutils.h>


using namespace rl::agents::dqn::trainers;

namespace seed_impl
{
    Trainer::Trainer(
        std::shared_ptr<rl::agents::dqn::Module> module,
        std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        std::shared_ptr<rl::env::Factory> env_factory,
        std::shared_ptr<rl::buffers::samplers::Uniform<rl::buffers::Tensor>> sampler,
        const SEEDOptions &options
    ) : options{options}
    {
        this->module = module;
        this->target_module = std::dynamic_pointer_cast<rl::agents::dqn::Module>(module->clone());
        this->value_parser = value_parser;
        this->optimizer = optimizer;
        this->env_factory = env_factory;
        this->sampler = sampler;
    }

    void Trainer::start() {
        running = true;
        training_thread = std::thread(&Trainer::worker, this);
    }

    void Trainer::stop() {
        running = false;
        if (training_thread.joinable()) {
            training_thread.join();
        }
    }

    void Trainer::worker()
    {
        torch::StreamGuard stream_guard{c10::cuda::getStreamFromPool()};
        auto period = std::chrono::seconds(options.checkpoint_callback_period_seconds);
        size_t i = 1;
        auto next_callback = std::chrono::high_resolution_clock::now() + period;

        while (running) {
            step();
            target_network_update();

            if (std::chrono::high_resolution_clock::now() >= next_callback) {
                if (options.checkpoint_callback) {
                    options.checkpoint_callback(i++ * options.checkpoint_callback_period_seconds);
                }
                next_callback = next_callback + period;
            }
        }
    }

    void Trainer::step()
    {
        auto sample_storage = sampler->sample(options.batch_size);
        const auto &sample{*sample_storage};

        auto outputs = module->forward(sample[0].to(options.network_device));
        auto masks = sample[1].to(options.network_device);

        torch::Tensor next_outputs;
        torch::Tensor next_actions;

        auto next_states = sample[5].to(options.network_device);
        auto next_masks = sample[6].to(options.network_device);
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
            sample[2].to(options.network_device),
            sample[3].to(options.network_device),
            sample[4].to(options.network_device),
            next_outputs,
            next_masks,
            next_actions,
            std::pow(options.discount, options.n_step)
        );

        loss = loss.mean();
        optimizer->zero_grad();
        loss.backward();
        auto grad_norm = rl::torchutils::compute_gradient_norm(optimizer);
        if (grad_norm.item().toFloat() > options.max_gradient_norm) {
            rl::torchutils::scale_gradients(optimizer, 1.0f / options.max_gradient_norm);
        }
        optimizer->step();

        if (options.logger) {
            options.logger->log_scalar("SEEDDQN/Loss", loss.item().toFloat());
            options.logger->log_scalar("SEEDDQN/Gradient norm", grad_norm.item().toFloat());
            options.logger->log_frequency("SEEDDQN/Update frequency", 1);
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
