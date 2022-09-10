#include "rl/agents/dqn/trainers/basic.h"

#include <rl/buffers/tensor.h>
#include <rl/buffers/samplers/uniform.h>
#include <rl/policies/constraints/categorical_mask.h>


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
        if (!options.double_dqn) {
            throw std::runtime_error{"Disabling double DQN is not yet implemented."};
        }
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

        size_t env_steps{0};
        size_t train_steps{0};

        auto execute_train_step = [&] () {
            auto sample_storage = sampler.sample(options.batch_size);
            auto &sample = *sample_storage;
            
            auto output = module->forward(sample[0]);
            rl::policies::constraints::CategoricalMask mask{sample[1]};
            output->apply_mask(mask);

            std::unique_ptr<rl::agents::dqn::modules::BaseOutput> next_output;
            
            {
                torch::InferenceMode guard{};
                rl::policies::constraints::CategoricalMask next_mask{sample[6]};
                next_output = target_module->forward(sample[5]);
                next_output->apply_mask(next_mask);
            }

            auto loss = output->loss(sample[2], sample[3], sample[4], *next_output, options.discount);
            optimizer->zero_grad();
            loss.backward();
            optimizer->step();

            if (options.logger) {
                options.logger->log_scalar("DQN/Loss", loss.item().toFloat());
            }

            train_steps++;
        };

        auto execute_env_step = [&] () {
            torch::InferenceMode guard{};
            if (env->is_terminal()) env->reset();
            auto state = env->state();
            auto output = module->forward(state->state.unsqueeze(0));
            const auto &mask = dynamic_cast<const CategoricalMask&>(*state->action_constraint);
            output->apply_mask(mask);
            
            auto policy = this->policy->policy(*output);
            auto action = policy->sample().squeeze(0);

            auto observation = env->step(action);
            
            const auto &next_mask = dynamic_cast<const CategoricalMask&>(*observation->state->action_constraint);

            buffer->add({
                state->state.unsqueeze(0).to(options.replay_device),
                mask.mask().unsqueeze(0).to(options.replay_device),
                action.unsqueeze(0).to(options.replay_device),
                torch::tensor(observation->reward, tensor_options[3]),
                torch::tensor(!observation->terminal, tensor_options[4]),
                observation->state->state.unsqueeze(0).to(options.replay_device),
                next_mask.mask().unsqueeze(0).to(options.replay_device)
            });

            env_steps++;
        };

        auto sync_modules = [&] () {
            auto target_parameters = target_module->parameters();
            auto parameters = module->parameters();

            for (int i = 0; i < parameters.size(); i++) {
                target_parameters[i].copy_(parameters[i]);
            }
        };

        auto stop_time = std::chrono::high_resolution_clock::now() + std::chrono::seconds{duration};
        while (std::chrono::high_resolution_clock::now() < stop_time)
        {
            if (buffer->size() < options.minimum_replay_buffer_size) {
                execute_env_step();
                continue;
            }

            if (env_steps % options.environment_steps_per_training_step == 0) {
                execute_train_step();

                if (train_steps % options.target_network_update_steps == 0) {
                    sync_modules();
                }
            }

            execute_env_step();
        }
    }
}
