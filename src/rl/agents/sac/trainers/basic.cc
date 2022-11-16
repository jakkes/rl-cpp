#include "rl/agents/sac/trainers/basic.h"

#include <rl/torchutils.h>


namespace F = torch::nn::functional;
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

    void Basic::initialize_buffer()
    {
        auto env = env_factory->get();
        auto state = env->reset();
        auto state_sizes = state->state.sizes();
        auto action = actor->forward(state->state.unsqueeze(0).to(options.network_device)).sample().squeeze(0);

        std::vector<std::vector<int64_t>> tensor_shapes{};
        tensor_shapes.push_back(state->state.sizes().vec());   // States
        tensor_shapes.push_back(action.sizes().vec());   // Actions
        tensor_shapes.push_back({});   // Rewards
        tensor_shapes.push_back({});   // Not terminals
        tensor_shapes.push_back(state->state.sizes().vec());   // Next states

        std::vector<torch::TensorOptions> tensor_options{};
        tensor_options.push_back(state->state.options().device(options.replay_device));
        tensor_options.push_back(torch::TensorOptions{}.device(options.replay_device));
        tensor_options.push_back(torch::TensorOptions{}.device(options.replay_device));
        tensor_options.push_back(torch::TensorOptions{}.dtype(torch::kBool).device(options.replay_device));
        tensor_options.push_back(state->state.options().device(options.replay_device));

        buffer = std::make_shared<rl::buffers::Tensor>(
            options.replay_buffer_size,
            tensor_shapes,
            tensor_options
        );
        sampler = std::make_shared<rl::buffers::samplers::Uniform<rl::buffers::Tensor>>(buffer);
    }

    torch::Tensor Basic::u_to_a(const torch::Tensor &u) {
        auto a = u.tanh();
        return 0.5f * (a + 1.0f) * (options.action_range_max - options.action_range_min) + options.action_range_min;
    }

    torch::Tensor Basic::a_to_u(const torch::Tensor &a) {
        return torch::arctanh(2.0f * (a - options.action_range_min) / (options.action_range_max - options.action_range_min) - 1.0f);
    }

    torch::Tensor Basic::log_pi_a(const torch::Tensor &u_, const ActorOutput &actor_output)
    {
        auto mean = actor_output.mean();
        auto std = actor_output.std();
        auto u = u_;

        assert(mean.sizes().size() > 0);
        if (mean.sizes().size() == 1) {
            mean = mean.unsqueeze(1);
            std = std.unsqueeze(1);
            u = u.unsqueeze(1);
        }
        assert(mean.sizes().size() == 2);
        assert(std.sizes().size() == 2);
        assert(u.sizes().size() == 2);

        auto log_mu = -0.5f * ((u - mean) / std).square_().sum(-1) - (2.5066282746310002f * std).log().sum(-1);
        auto out = log_mu - (1.0f - u.tanh().square()).log().sum(-1);
        return out;
    }

    void Basic::init_env() {
        env = env_factory->get();
    }

    void Basic::execute_env_step()
    {
        torch::InferenceMode guard{};
        bool should_log_start_value{false};
        // If environment is in terminal state, or this is the first step ever.
        // (Note, env may not be in a terminal state on the first call due to buffer
        // initialization.)
        if (env->is_terminal() || env_steps == 0) {
            env->reset();
            should_log_start_value = true;
        }

        std::shared_ptr<rl::env::State> state = env->state();
        if (state->action_constraint->is_type<rl::policies::constraints::Empty>()) {}
        else if (state->action_constraint->is_type<rl::policies::constraints::Box>()) {
            auto &box = state->action_constraint->as_type<rl::policies::constraints::Box>();
            if (
                !box.upper_bound().le(options.action_range_max).all().item().toBool()
                || !box.lower_bound().ge(options.action_range_min).all().item().toBool()
            ) {
                throw std::runtime_error{"Action constraint not fulfilled."};
            }
        }
        else {
            throw std::runtime_error{"Unsupported action constraint, WIP..."};
        }

        auto output = actor->forward( state->state.unsqueeze(0).to(options.network_device) );
        
        if (options.logger && should_log_start_value) {
            options.logger->log_scalar("SAC/StartValue", output.value().item().toFloat());
        }

        auto u = output.sample().squeeze(0);
        auto a = u_to_a(u);
        
        auto observation = env->step(a.to(options.environment_device));
        
        observation->state->action_constraint->to(options.replay_device);
        state->action_constraint->to(options.replay_device);

        buffer->add(
            {
                state->state.unsqueeze(0).to(options.replay_device),
                a.unsqueeze(0).to(options.replay_device),
                torch::tensor({observation->reward}, buffer->tensor_options()[2]),
                torch::tensor({!observation->terminal}, buffer->tensor_options()[3]),
                observation->state->state.unsqueeze(0).to(options.replay_device)
            }
        );

        if (observation->terminal && options.logger) {
            options.logger->log_scalar("SAC/EndValue", output.value().item().toFloat());
        }

        env_steps++;
    }

    void Basic::execute_train_step()
    {
        auto sample_storage = sampler->sample(options.batch_size);
        auto &sample = *sample_storage;

        torch::Tensor value_loss, policy_loss;
        std::vector<torch::Tensor> critic_losses{}; critic_losses.reserve(critics.size());
        auto current_policy = actor->forward(sample[0].to(options.network_device));

        // Compute value loss
        {
            torch::Tensor critic_outputs, entropy_estimator;
            {
                torch::NoGradGuard guard{};
                auto u = current_policy.sample().detach();
                auto a = u_to_a(u);
                entropy_estimator = - log_pi_a(u, current_policy);

                critic_outputs = critics[0]->forward(sample[0].to(options.network_device), a).value();
                for (int i = 0; i < critics.size(); i++) {
                    critic_outputs = critic_outputs.min(critics[i]->forward(sample[0].to(options.network_device), a).value());
                }
            }
            value_loss = F::huber_loss(
                current_policy.value(),
                critic_outputs + options.temperature * entropy_estimator,
                F::HuberLossFuncOptions{}.reduction(torch::kNone).delta(options.huber_loss_delta)
            );
            assert (!value_loss.isnan().any().item().toBool());
        }

        // Compute critic losses
        {
            torch::Tensor target;
            {
                torch::NoGradGuard guard{};
                auto next_policy = actor_target->forward(sample[4].to(options.network_device));
                target = sample[2].to(options.network_device) + options.discount * sample[3].to(options.network_device) * next_policy.value();
            }

            for (int i = 0; i < critics.size(); i++) {
                auto value = critics[i]->forward(sample[0].to(options.network_device), sample[1].to(options.network_device)).value();
                critic_losses.push_back(
                    F::huber_loss(
                        value,
                        target,
                        F::HuberLossFuncOptions{}.reduction(torch::kNone).delta(options.huber_loss_delta)
                    )
                );

                assert(!critic_losses[i].isnan().any().item().toBool());
            }
        }

        // Compute policy loss
        {
            auto u = current_policy.sample();
            auto a = u_to_a(u);
            torch::Tensor critic_outputs = critics[0]->forward(sample[0].to(options.network_device), a).value();
            for (int i = 0; i < critics.size(); i++) {
                critic_outputs = critic_outputs.min(critics[i]->forward(sample[0].to(options.network_device), a).value());
            }
            policy_loss = log_pi_a(u, current_policy) - critic_outputs;
            assert(!policy_loss.isnan().any().item().toBool());
        }

        auto mean_policy_loss = policy_loss.mean();
        auto mean_value_loss = value_loss.mean();
        auto actor_loss = mean_policy_loss + mean_value_loss;
        std::vector<torch::Tensor> mean_critic_losses{};
        mean_critic_losses.reserve(critics.size());
        for (int i = 0; i < critics.size(); i++) {
            mean_critic_losses.push_back(critic_losses[i].mean());
        }

        // Apply backward passes
        actor_optimizer->zero_grad();
        actor_loss.backward();
        auto actor_grad_norm = rl::torchutils::compute_gradient_norm(actor_optimizer).item().toFloat();
        if (actor_grad_norm > options.max_gradient_norm) {
            rl::torchutils::scale_gradients(actor_optimizer, 1.0f / actor_grad_norm);
        }
        actor_optimizer->step();

        std::vector<float> critic_grad_norms{}; critic_grad_norms.resize(critics.size());
        for (int i = 0; i < critics.size(); i++)
        {
            critic_optimizers[i]->zero_grad();
            mean_critic_losses[i].backward();
            critic_grad_norms[i] = rl::torchutils::compute_gradient_norm(critic_optimizers[i]).item().toFloat();
            if (critic_grad_norms[i] > options.max_gradient_norm) {
                rl::torchutils::scale_gradients(critic_optimizers[i], 1.0f / critic_grad_norms[i]);
            }
            critic_optimizers[i]->step();
        }

        if (options.logger) {
            options.logger->log_scalar("SAC/PolicyLoss", mean_policy_loss.item().toFloat());
            options.logger->log_scalar("SAC/ValueLoss", mean_value_loss.item().toFloat());
            for (int i = 0; i < critics.size(); i++) {
                options.logger->log_scalar("SAC/CriticLoss" + std::to_string(i), mean_critic_losses[i].item().toFloat());
            }

            options.logger->log_scalar("SAC/ActorGradNorm", actor_grad_norm);
            for (int i = 0; i < critics.size(); i++) {
                options.logger->log_scalar("SAC/CriticGradNorm" + std::to_string(i), critic_grad_norms[i]);
            }
        }

        {
            torch::NoGradGuard guard{};
            auto target_parameters = actor_target->parameters();
            auto parameters = actor->parameters();

            for (int i = 0; i < parameters.size(); i++) {
                target_parameters[i].add_(options.target_network_lr * (parameters[i] - target_parameters[i]));
            }

            train_steps++;
        }
    }

    void Basic::run(size_t duration)
    {
        init_env();
        initialize_buffer();

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
