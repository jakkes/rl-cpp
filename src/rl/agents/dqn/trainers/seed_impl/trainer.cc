#include "trainer.h"


using namespace rl::agents::dqn::trainers;

namespace seed_impl
{
    Trainer::Trainer(
        std::shared_ptr<rl::agents::dqn::modules::Base> module,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        std::shared_ptr<rl::env::Factory> env_factory,
        std::shared_ptr<rl::buffers::samplers::Uniform<rl::buffers::Tensor>> sampler,
        const SEEDOptions &options
    ) : options{options}
    {
        this->module = module;
        this->target_module = module->clone();
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

    void Trainer::worker() {
        while (running) {
            step();
        }
    }

    void Trainer::step()
    {
        auto sample_storage = sampler->sample(options.batch_size);
        const auto &sample{*sample_storage};

        auto output = module->forward(sample[0].to(options.network_device));
        output->apply_mask(sample[1].to(options.network_device));

        std::unique_ptr<rl::agents::dqn::modules::BaseOutput> next_output;
        torch::Tensor next_actions;

        {
            torch::InferenceMode guard{};
            auto next_state = sample[5].to(options.network_device);
            auto next_mask = sample[6].to(options.network_device);
            next_output = target_module->forward(next_state);
            next_output->apply_mask(next_mask);

            if (options.double_dqn) {
                auto tmp_output = module->forward(next_state);
                tmp_output->apply_mask(next_mask);
                next_actions = tmp_output->greedy_action();
            } else {
                next_actions = next_output->greedy_action();
            }
        }

        auto loss = output->loss(
            sample[2].to(options.network_device),
            sample[3].to(options.network_device),
            sample[4].to(options.network_device),
            *next_output,
            next_actions,
            std::pow(options.discount, options.n_step)
        );
        loss = loss.mean();
        optimizer->zero_grad();
        loss.backward();
        auto grad_norm = compute_gradient_norm();
        optimizer->step();

        if (options.logger) {
            options.logger->log_scalar("DQN/Loss", loss.item().toFloat());
            options.logger->log_scalar("DQN/Gradient norm", grad_norm.item().toFloat());
            options.logger->log_frequency("DQN/Update frequency", 1);
        }
    }
}