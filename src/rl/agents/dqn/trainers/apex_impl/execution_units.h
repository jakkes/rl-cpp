#ifndef RL_AGENTS_DQN_TRAINERS_APEX_IMPL_EXECUTION_UNITS_H_
#define RL_AGENTS_DQN_TRAINERS_APEX_IMPL_EXECUTION_UNITS_H_

#include <memory>

#include <torch/torch.h>
#include <rl/torchutils/execution_unit.h>
#include <rl/torchutils/gradient_norm.h>
#include <rl/torchutils/scale_gradients.h>
#include <rl/agents/dqn/module.h>
#include <rl/agents/dqn/value_parsers/base.h>
#include <rl/agents/dqn/trainers/apex.h>


namespace rl::agents::dqn::trainers::apex_impl
{
    class InferenceUnit : public rl::torchutils::ExecutionUnit
    {
        public:
            InferenceUnit(
                std::shared_ptr<rl::agents::dqn::Module> module,
                std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser,
                const ApexOptions &options
            ) : 
                rl::torchutils::ExecutionUnit{
                    options.worker_batchsize, options.network_device, options.enable_inference_cuda_graph
                },
                module{module},
                value_parser{value_parser}
            {}

        private:
            rl::torchutils::ExecutionUnitOutput forward(const std::vector<torch::Tensor> &inputs) override
            {
                torch::InferenceMode guard{};
                auto &states = inputs[0];
                auto &masks = inputs[1];

                rl::torchutils::ExecutionUnitOutput out{1, 0};
                out.tensors[0] = value_parser->values(module->forward(states), masks);
                return out;
            }

        private:
            std::shared_ptr<rl::agents::dqn::Module> module;
            std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser;
    };

    class TrainingUnit : public rl::torchutils::ExecutionUnit
    {
        public:
            TrainingUnit(
                std::shared_ptr<rl::agents::dqn::Module> module,
                std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                const ApexOptions &options
            ) : 
                rl::torchutils::ExecutionUnit{
                    options.batch_size, options.network_device, options.enable_training_cuda_graph
                }, options{options}
            {
                this->module = module;
                this->target_module = std::dynamic_pointer_cast<rl::agents::dqn::Module>(module->clone());
                this->value_parser = value_parser;
                this->optimizer = optimizer;
            }

        private:
            rl::torchutils::ExecutionUnitOutput forward(const std::vector<torch::Tensor> &samples) override
            {
                auto outputs = module->forward(samples[0]);
                auto masks = samples[1];

                torch::Tensor next_actions;
                torch::Tensor next_outputs;
                auto next_masks = samples[6];
                auto next_states = samples[5];
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
                    samples[2],
                    samples[3],
                    samples[4],
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

                rl::torchutils::ExecutionUnitOutput out{0, 2};
                out.scalars[0] = loss.detach();
                out.scalars[1] = grad_norm.detach();

                {
                    torch::InferenceMode guard{};
                    auto target_parameters = target_module->parameters();
                    auto parameters = module->parameters();

                    for (int i = 0; i < parameters.size(); i++) {
                        target_parameters[i].add_(parameters[i] - target_parameters[i], options.target_network_lr);
                    }
                }

                return out;
            }

        private:
            const ApexOptions options;
            std::shared_ptr<rl::agents::dqn::Module> module;
            std::shared_ptr<rl::agents::dqn::Module> target_module;
            std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser;
            std::shared_ptr<torch::optim::Optimizer> optimizer;
    };
}

#endif /* RL_AGENTS_DQN_TRAINERS_APEX_IMPL_EXECUTION_UNITS_H_ */
