#include "rl/agents/sac/trainers/basic.h"


#include <rl/buffers/tensor.h>
#include <rl/buffers/samplers/uniform.h>

namespace rl::agents::sac::trainers
{

    Basic::Basic(
        std::shared_ptr<rl::agents::sac::Actor> actor,
        std::vector<std::shared_ptr<rl::agents::sac::Critic>> critics,
        std::shared_ptr<torch::optim::Optimizer> actor_optimizer,
        std::vector<std::shared_ptr<torch::optim::Optimizer>> critic_optimizers,
        std::shared_ptr<rl::env::Factory> env_factory,
        const BasicOptions &options
    ) : 
        options{options},
        actor{actor},
        actor_target{actor->clone()},
        critics{critics},
        actor_optimizer{actor_optimizer},
        critic_optimizers{critic_optimizers},
        env_factory{env_factory}
    {}

    void Basic::run(size_t duration)
    {
        auto env = env_factory->get();
        auto state = env->reset();
        auto state_sizes = state->state.sizes();

        std::vector<std::vector<int64_t>> tensor_shapes{};
        tensor_shapes.push_back(state->state.sizes().vec());   // States
        tensor_shapes.push_back({});   // Actions
        tensor_shapes.push_back({});   // Rewards
        tensor_shapes.push_back({});   // Not terminals
        tensor_shapes.push_back(state->state.sizes().vec());   // Next states

        std::vector<torch::TensorOptions> tensor_options{};
        tensor_options.push_back(state->state.options().device(options.replay_device));
        tensor_options.push_back(torch::TensorOptions{}.device(options.replay_device));
        tensor_options.push_back(torch::TensorOptions{}.device(options.replay_device));
        tensor_options.push_back(torch::TensorOptions{}.dtype(torch::kBool).device(options.replay_device));
        tensor_options.push_back(state->state.options().device(options.replay_device));

        auto buffer = std::make_shared<rl::buffers::Tensor>(
            options.replay_buffer_size,
            tensor_shapes,
            tensor_options
        );

        rl::buffers::samplers::Uniform sampler{buffer};

        size_t env_steps{0};
        size_t train_steps{0};

        auto compute_gradient_norm = [this] (std::shared_ptr<torch::optim::Optimizer> optimizer) {
            auto grad_norm = torch::tensor(0.0f, torch::TensorOptions{}.device(options.network_device));

            for (const auto &param_group : optimizer->param_groups()) {
                for (const auto &param : param_group.params()) {
                    grad_norm += param.grad().square().sum();
                }
            }

            return grad_norm.sqrt_();
        };

        auto u_to_a = [&] (const torch::Tensor &u) {
            auto a = u.tanh();
            return 0.5f * (a + 1.0f) * (options.action_range_max - options.action_range_min) + options.action_range_min;
        };

        auto a_to_u = [&] (const torch::Tensor &a) {
            return torch::arctanh(2.0f * (a - options.action_range_min) / (options.action_range_max - options.action_range_min) - 1.0f);
        };

        auto log_pi_u = [&] (const torch::Tensor &u, const ActorOutput &actor) {
            auto log_mu = torch::sum(-0.5 * (u - actor.mean()).square_() / actor.variance() - 0.5 * actor.variance().log() - 0.39908993417f, -1);
            return log_mu - (1.0f - u.tanh().square_()).log_().sum(-1);
        };

        auto execute_env_step = [&] () {
            torch::InferenceMode guard{};
            bool should_log_start_value{false};
            // If environment is in terminal state, or this is the first step.
            if (env->is_terminal() || env_steps == 0) {
                env->reset();
                should_log_start_value = true;
            }
            std::shared_ptr<rl::env::State> state = env->state();
            if (!state->action_constraint->is_type<rl::policies::constraints::Empty>()) {
                throw std::runtime_error{"SAC only supports empty policy constraints. WIP..."};
            }

            auto output = actor->forward( state->state.unsqueeze(0).to(options.network_device) );
            
            if (options.logger && should_log_start_value) {
                options.logger->log_scalar("SAC/StartValue", output->value().item().toFloat());
            }

            auto u = output->sample().squeeze(0);
            auto a = u_to_a(u);
            
            auto observation = env->step(a.to(options.environment_device));
            
            observation->state->action_constraint->to(options.replay_device);
            state->action_constraint->to(options.replay_device);

            buffer->add(
                {
                    state->state.to(options.replay_device),
                    a.to(options.replay_device),
                    torch::tensor({observation->reward}, tensor_options[2]),
                    torch::tensor({!observation->terminal}, tensor_options[3]),
                    observation->state->state.to(options.replay_device)
                }
            );

            if (observation->terminal && options.logger) {
                options.logger->log_scalar("SAC/EndValue", output->value().item().toFloat());
            }

            env_steps++;
        };

        auto execute_train_step = [&] () {
            auto sample_storage = sampler.sample(options.batch_size);
            auto &sample = *sample_storage;

            torch::Tensor value_loss, policy_loss;
            std::vector<torch::Tensor> critic_losses{}; critic_losses.reserve(critics.size());
            auto current_policy = actor->forward(sample[0]);

            // Compute value loss
            {
                torch::Tensor actions, critic_outputs, entropy_estimator;
                {
                    torch::InferenceMode guard{};
                    auto u = current_policy->sample().detach();
                    auto a = u_to_a(u);
                    entropy_estimator = - log_pi_u(u, *current_policy);

                    critic_outputs = critics[0]->forward(sample[0], a)->value();
                    for (int i = 0; i < critics.size(); i++) {
                        critic_outputs = critic_outputs.min(critics[i]->forward(sample[0], a)->value());
                    }
                }
                value_loss = (current_policy->value() - (critic_outputs + options.temperature * entropy_estimator)).square_();
            }

            // Compute critic losses
            {
                std::unique_ptr<rl::agents::sac::ActorOutput> next_policy;
                {
                    torch::InferenceMode guard{};
                    next_policy = actor_target->forward(sample[4]);
                }

                for (int i = 0; i < critics.size(); i++) {
                    auto target = sample[2] + options.discount * sample[3] * next_policy->value();
                    auto value = critics[i]->forward(sample[0], sample[1])->value();
                    critic_losses.push_back( (target - value).square_().mean() );
                }
            }

            // Compute policy loss
            {
                auto u = current_policy->sample();
                auto a = u_to_a(u);
                torch::Tensor critic_outputs = critics[0]->forward(sample[0], a)->value();
                for (int i = 0; i < critics.size(); i++) {
                    critic_outputs = critic_outputs.min(critics[i]->forward(sample[0], a)->value());
                }
                policy_loss = log_pi_u(u, *current_policy) - critic_outputs;
            }

            auto mean_policy_loss = policy_loss.mean();
            auto mean_value_loss = value_loss.mean();
            std::vector<torch::Tensor> mean_critic_losses{critics.size()};
            for (int i = 0; i < critics.size(); i++) {
                mean_critic_losses.push_back(critic_losses[i].mean());
            }

            // Apply backward passes
            actor_optimizer->zero_grad();
            mean_policy_loss.backward();
            mean_value_loss.backward();
            actor_optimizer->step();

            for (int i = 0; i < critics.size(); i++)
            {
                critic_optimizers[i]->zero_grad();
                mean_critic_losses[i].backward();
                critic_optimizers[i]->step();
            }

            if (options.logger) {
                options.logger->log_scalar("SAC/PolicyLoss", mean_policy_loss.item().toFloat());
                options.logger->log_scalar("SAC/ValueLoss", value_loss.item().toFloat());
                for (int i = 0; i < critics.size(); i++) {
                    options.logger->log_scalar("SAC/CriticLoss" + std::to_string(i), mean_critic_losses[i].item().toFloat());
                }

                options.logger->log_scalar("SAC/ActorGradNorm", compute_gradient_norm(actor_optimizer).item().toFloat());
                for (int i = 0; i < critics.size(); i++) {
                    options.logger->log_scalar("SAC/CriticGradNorm" + std::to_string(i), compute_gradient_norm(critic_optimizers[i]).item().toFloat());
                }
            }

            train_steps++;
        };

        auto sync_actor_modules = [&] () {
            torch::InferenceMode guard{};
            auto target_parameters = actor_target->parameters();
            auto parameters = actor->parameters();

            for (int i = 0; i < parameters.size(); i++) {
                target_parameters[i].add_(options.target_network_lr * (parameters[i] - target_parameters[i]));
            }
        };

        auto stop_time = std::chrono::high_resolution_clock::now() + std::chrono::seconds{duration};

        while (std::chrono::high_resolution_clock::now() < stop_time && buffer->size() < options.minimum_replay_buffer_size) {
            execute_env_step();
            continue;
        }

        env_steps = 0;

        while (std::chrono::high_resolution_clock::now() < stop_time)
        {
            if (env_steps > options.environment_steps_per_training_step * train_steps) {
                execute_train_step();
                sync_actor_modules();
                if (train_steps % options.checkpoint_callback_period == 0) {
                    if (options.checkpoint_callback) options.checkpoint_callback(train_steps);
                }
            }
            else {
                execute_env_step();
            }
        }
    }
}
