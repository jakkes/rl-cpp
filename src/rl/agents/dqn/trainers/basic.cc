#include "rl/agents/dqn/trainers/basic.h"

#include <rl/buffers/tensor.h>
#include <rl/buffers/samplers/uniform.h>
#include <rl/policies/constraints/categorical_mask.h>
#include <rl/utils/n_step_collector.h>


using rl::policies::constraints::CategoricalMask;

namespace rl::agents::dqn::trainers
{

    Basic::Basic(
        std::shared_ptr<rl::agents::dqn::modules::Base> module,
        std::shared_ptr<rl::agents::dqn::policies::Base> policy,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        std::shared_ptr<rl::env::Factory> env_factory,
        const BasicOptions &options
    ) :
        module{module},
        target_module{module->clone()},
        policy{policy},
        optimizer{optimizer},
        env_factory{env_factory},
        options{options}
    {

    }

    void Basic::run(size_t duration)
    {
        auto env = env_factory->get();
        auto state = env->reset();
        auto state_sizes = state->state.sizes();
        const CategoricalMask &mask_constraint = dynamic_cast<const CategoricalMask&>(*state->action_constraint);
        
        std::vector<std::vector<int64_t>> tensor_shapes{};
        tensor_shapes.push_back(state->state.sizes().vec());   // States
        tensor_shapes.push_back(mask_constraint.mask().sizes().vec()); // Masks
        tensor_shapes.push_back({});   // Actions
        tensor_shapes.push_back({});   // Rewards
        tensor_shapes.push_back({});   // Not terminals
        tensor_shapes.push_back(state->state.sizes().vec());   // Next states
        tensor_shapes.push_back(mask_constraint.mask().sizes().vec()); // Next masks

        std::vector<torch::TensorOptions> tensor_options{};
        tensor_options.push_back(state->state.options().device(options.replay_device));
        tensor_options.push_back(mask_constraint.mask().options().device(options.replay_device));
        tensor_options.push_back(torch::TensorOptions{}.dtype(torch::kLong).device(options.replay_device));
        tensor_options.push_back(torch::TensorOptions{}.dtype(torch::kFloat32).device(options.replay_device));
        tensor_options.push_back(torch::TensorOptions{}.dtype(torch::kBool).device(options.replay_device));
        tensor_options.push_back(state->state.options().device(options.replay_device));
        tensor_options.push_back(mask_constraint.mask().options().device(options.replay_device));

        auto buffer = std::make_shared<rl::buffers::Tensor>(
            options.replay_buffer_size,
            tensor_shapes,
            tensor_options
        );

        rl::buffers::samplers::Uniform sampler{buffer};
        rl::utils::NStepCollector collector{options.n_step, options.discount};
        std::unique_ptr<rl::agents::dqn::utils::HindsightReplayEpisode> episode;

        size_t env_steps{0};
        size_t train_steps{0};

        auto compute_gradient_norm = [&] () {
            auto grad_norm = torch::tensor(0.0f, torch::TensorOptions{}.device(options.network_device));

            for (const auto &param_group : optimizer->param_groups()) {
                for (const auto &param : param_group.params()) {
                    grad_norm += param.grad().square().sum();
                }
            }

            return grad_norm.sqrt_();
        };

        auto add_transitions = [&] (const std::vector<rl::utils::NStepCollectorTransition> &transitions) {
            for (const auto &transition : transitions) {
                auto mask = dynamic_cast<const CategoricalMask&>(*transition.state->action_constraint).mask();
                auto next_mask = dynamic_cast<const CategoricalMask&>(*transition.next_state->action_constraint).mask();
                buffer->add({
                    transition.state->state.unsqueeze(0).to(options.replay_device),
                    mask.unsqueeze(0).to(options.replay_device),
                    transition.action.unsqueeze(0).to(options.replay_device),
                    torch::tensor({transition.reward}, tensor_options[3]),
                    torch::tensor({!transition.terminal}, tensor_options[4]),
                    transition.next_state->state.unsqueeze(0).to(options.replay_device),
                    next_mask.unsqueeze(0).to(options.replay_device)
                });
            }
        };

        auto add_hindsight_replay = [&] () {
            if (!options.hindsight_replay_callback) return;

            auto should_add = options.hindsight_replay_callback(episode.get());
            if (!should_add) return;

            for (int i = 0; i < episode->actions.size(); i++) {
                auto state = std::make_shared<rl::env::State>();
                state->state = episode->states[i];
                state->action_constraint = std::make_shared<rl::policies::constraints::CategoricalMask>(episode->masks[i]);
                auto transitions = collector.step(state, episode->actions[i], episode->rewards[i], i == episode->actions.size() - 1);
                add_transitions(transitions);
            }
        };

        auto execute_train_step = [&] () {
            auto sample_storage = sampler.sample(options.batch_size);
            auto &sample = *sample_storage;
            
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

            {
                torch::NoGradGuard guard{};
                auto target_parameters = module->parameters();
                auto parameters = target_module->parameters();

                for (int i = 0; i < parameters.size(); i++) {
                    target_parameters[i].add_(options.target_network_lr * (parameters[i] - target_parameters[i]));
                }
            }

            train_steps++;
        };

        auto execute_env_step = [&] () {
            torch::InferenceMode guard{};
            bool should_log_start_value{false};

            // If environment is in terminal state, or this is the first step.
            if (env->is_terminal() || env_steps == 0) {
                auto state = env->reset();
                should_log_start_value = true;
                episode = std::make_unique<rl::agents::dqn::utils::HindsightReplayEpisode>();
            }
            std::shared_ptr<rl::env::State> state = env->state();
            auto output = module->forward(state->state.unsqueeze(0).to(options.network_device));
            auto mask = dynamic_cast<const CategoricalMask&>(*state->action_constraint).mask();
            output->apply_mask(mask.unsqueeze(0).to(options.network_device));

            episode->states.push_back(state->state.clone());
            episode->masks.push_back(mask.clone());

            if (options.logger && should_log_start_value) {
                options.logger->log_scalar("DQN/StartValue", output->value().max().item().toFloat());
            }
            
            auto policy = this->policy->policy(*output);
            policy->include(state->action_constraint);

            auto action = policy->sample().squeeze(0);
            episode->actions.push_back(action.clone());

            auto observation = env->step(action.to(options.environment_device));
            episode->rewards.push_back(observation->reward);

            auto transitions = collector.step(
                state,
                action,
                observation->reward,
                observation->terminal
            );

            add_transitions(transitions);

            if (observation->terminal) {
                episode->states.push_back(observation->state->state.clone());
                episode->masks.push_back(dynamic_cast<const CategoricalMask&>(*observation->state->action_constraint).mask().clone());
                add_hindsight_replay();
                
                if (options.logger) {
                    options.logger->log_scalar("DQN/EndValue", output->value().max().item().toFloat());
                }
            }

            env_steps++;
        };

        auto stop_time = std::chrono::high_resolution_clock::now() + std::chrono::seconds{duration};
        while (buffer->size() < options.minimum_replay_buffer_size
                            && std::chrono::high_resolution_clock::now() < stop_time)
        {
            execute_env_step();
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
