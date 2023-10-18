#include "rl/agents/dqn/trainers/basic.h"

#include <rl/torchutils/execution_unit.h>
#include <rl/buffers/tensor.h>
#include <rl/buffers/samplers/uniform.h>
#include <rl/policies/constraints/categorical_mask.h>
#include <rl/utils/reward/n_step_collector.h>
#include <rl/torchutils/torchutils.h>


using rl::policies::constraints::CategoricalMask;
using namespace torch::indexing;
using namespace rl::torchutils;


namespace rl::agents::dqn::trainers
{
    class TrainUnit : public ExecutionUnit
    {
        public:
            TrainUnit(
                std::shared_ptr<rl::agents::dqn::Module> module,
                std::shared_ptr<rl::agents::dqn::Module> target_module,
                std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                const BasicOptions &options
            ) : ExecutionUnit(options.batch_size, options.network_device, options.enable_cuda_graph),
                options{options}
            {
                this->module = module;
                this->target_module = target_module;
                this->value_parser = value_parser;
                this->optimizer = optimizer;
            }

        private:
            ExecutionUnitOutput forward(const std::vector<torch::Tensor> &sample)
            {
                auto output = module->forward(sample[0]);
                auto masks = sample[1];

                torch::Tensor next_outputs;
                torch::Tensor next_masks;
                torch::Tensor next_actions;
                
                {
                    torch::InferenceMode guard{};
                    auto next_state = sample[5];
                    next_masks = sample[6];
                    next_outputs = target_module->forward(next_state);

                    if (options.double_dqn) {
                        auto tmp_output = module->forward(next_state);
                        next_actions = value_parser->values(tmp_output, next_masks).argmax(-1);
                    } else {
                        next_actions = this->value_parser->values(next_state, next_masks).argmax(-1);
                    }
                }

                auto loss = value_parser->loss(
                    output,
                    masks,
                    sample[2],
                    sample[3],
                    sample[4],
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

                {
                    torch::NoGradGuard guard{};
                    auto target_parameters = target_module->parameters();
                    auto parameters = module->parameters();

                    for (int i = 0; i < parameters.size(); i++) {
                        target_parameters[i].add_(parameters[i] - target_parameters[i], options.target_network_lr);
                    }
                }

                ExecutionUnitOutput out{0, 2};
                out.scalars[0] = loss.detach();
                out.scalars[1] = grad_norm;

                return out;
            }

        private:
            const BasicOptions options;
            std::shared_ptr<rl::agents::dqn::Module> module;
            std::shared_ptr<rl::agents::dqn::Module> target_module;
            std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser;
            std::shared_ptr<torch::optim::Optimizer> optimizer;
    };


    class InferenceUnit : public ExecutionUnit
    {
        public:
            InferenceUnit(
                std::shared_ptr<rl::agents::dqn::Module> module,
                const BasicOptions &options
            ) : ExecutionUnit(1, options.network_device, options.enable_cuda_graph)
            {
                this->module = module;
            }

        private:
            ExecutionUnitOutput forward(const std::vector<torch::Tensor> &inputs) override
            {
                torch::InferenceMode guard{};
                ExecutionUnitOutput out{1, 0};
                out.tensors[0] = module->forward(inputs[0]);
                return out;
            }

        private:
            std::shared_ptr<rl::agents::dqn::Module> module;
            const BasicOptions options;
    };


    Basic::Basic(
        std::shared_ptr<rl::agents::dqn::Module> module,
        std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser,
        std::shared_ptr<rl::agents::dqn::policies::Base> policy,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        std::shared_ptr<rl::env::Factory> env_factory,
        const BasicOptions &options
    ) :
        module{module},
        target_module{std::dynamic_pointer_cast<rl::agents::dqn::Module>(module->clone())},
        value_parser{value_parser},
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
        rl::utils::reward::NStepCollector collector{options.n_step, options.discount};
        std::unique_ptr<rl::agents::dqn::utils::HindsightReplayEpisode> episode;

        TrainUnit train_unit{module, target_module, value_parser, optimizer, options};
        InferenceUnit inference_unit{module, options};

        size_t env_steps{0};
        size_t train_steps{0};

        auto add_transitions = [&] (const std::vector<rl::utils::reward::NStepCollectorTransition> &transitions) {
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
                auto transitions = collector.step(episode->states[i], episode->actions[i], episode->rewards[i], i == episode->actions.size() - 1);
                add_transitions(transitions);
            }
        };

        auto execute_train_step = [&] () {
            auto sample_storage = sampler.sample(options.batch_size);
            auto &sample = *sample_storage;

            auto output = train_unit({
                sample[0].to(options.network_device),
                sample[1].to(options.network_device),
                sample[2].to(options.network_device),
                sample[3].to(options.network_device),
                sample[4].to(options.network_device),
                sample[5].to(options.network_device),
                sample[6].to(options.network_device),
            });

            if (options.logger) {
                options.logger->log_scalar("DQN/Loss", output.scalars[0].item().toFloat());
                options.logger->log_scalar("DQN/Gradient norm", output.scalars[1].item().toFloat());
                options.logger->log_frequency("DQN/Update frequency", 1);
            }

            train_steps++;
        };

        auto execute_env_step = [&] () {
            bool should_log_start_value{false};

            // If environment is in terminal state, or this is the first step.
            if (env->is_terminal() || env_steps == 0) {
                auto state = env->reset();
                should_log_start_value = true;
                episode = std::make_unique<rl::agents::dqn::utils::HindsightReplayEpisode>();
            }
            std::shared_ptr<rl::env::State> state = env->state();

            auto inference_unit_output = inference_unit({state->state.unsqueeze(0).to(options.network_device)});
            auto &output = inference_unit_output.tensors[0];
            auto mask = dynamic_cast<const CategoricalMask&>(*state->action_constraint).mask().unsqueeze(0).to(options.network_device);

            assert(!output.requires_grad());
            
            auto values = value_parser->values(output, mask);
            episode->states.push_back(state);

            if (options.logger && should_log_start_value) {
                auto max_value = values.max().item().toFloat();
                auto min_value = values.where(~values.isneginf(), max_value).min().item().toFloat();
                options.logger->log_scalar("DQN/StartValue", max_value);
                options.logger->log_scalar("DQN/StartAdvantage", max_value - min_value);
            }

            auto policy = this->policy->policy(values, mask);
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
                episode->states.push_back(observation->state);
                add_hindsight_replay();
                
                if (options.logger) {
                    auto max_value = values.max().item().toFloat();
                    auto min_value = values.where(~values.isneginf(), max_value).min().item().toFloat();
                    options.logger->log_scalar("DQN/EndValue", max_value);
                    options.logger->log_scalar("DQN/EndAdvantage", max_value - min_value);
                }
            }

            env_steps++;
        };

        auto stop_time = std::chrono::high_resolution_clock::now() + std::chrono::seconds{duration};
        while (buffer->size() < options.minimum_replay_buffer_size
                            && std::chrono::high_resolution_clock::now() < stop_time)
        {
            execute_env_step();
            if (options.logger) {
                options.logger->log_scalar("DQN/BufferSize", buffer->size());
            }
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
                if (options.logger) {
                    options.logger->log_scalar("DQN/BufferSize", buffer->size());
                }
            }
        }
    }
}
