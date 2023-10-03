#ifndef RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_INFERENCE_UNIT_H_
#define RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_INFERENCE_UNIT_H_


#include <rl/agents/alpha_zero/alpha_zero.h>
#include <rl/torchutils/execution_unit.h>
#include <rl/torchutils/torchutils.h>


using namespace rl::agents::alpha_zero;

namespace trainer_impl
{
    class InferenceUnit : public rl::torchutils::ExecutionUnit
    {
        public:
            InferenceUnit(
                int max_batchsize,
                torch::Device device,
                std::shared_ptr<modules::Base> module,
                bool use_cuda_graph
            ) : rl::torchutils::ExecutionUnit(max_batchsize, device, use_cuda_graph), module{module}
            {}
        
            rl::torchutils::ExecutionUnitOutput forward(const std::vector<torch::Tensor> &inputs)
            {
                torch::NoGradGuard no_grad_guard{};
                auto module_output = module->forward(inputs[0]);
                auto policy_output = module_output->policy().get_probabilities();
                auto value_output = module_output->value_estimates();

                rl::torchutils::ExecutionUnitOutput out{2, 0};
                out.tensors[0] = policy_output;
                out.tensors[1] = value_output;

                return out;
            }

        private:
            std::shared_ptr<modules::Base> module;
    };

    class TrainingUnit : public rl::torchutils::ExecutionUnit
    {
        public:
            TrainingUnit(
                int max_batchsize,
                torch::Device device,
                std::shared_ptr<modules::Base> module,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                bool use_cuda_graph
            ) : rl::torchutils::ExecutionUnit(
                    max_batchsize, device, use_cuda_graph
                ),
                module{module},
                optimizer{optimizer}
            {
                if (use_cuda_graph) {
                    throw std::runtime_error{"CUDAGraph for training not yet supported."};
                }
            }

            rl::torchutils::ExecutionUnitOutput forward(const std::vector<torch::Tensor> &inputs)
            {
                auto &states = inputs[0];
                auto &posteriors = inputs[1];
                auto &rewards = inputs[2];
                auto module_output = module->forward(states);
                auto policy_loss = module_output->policy_loss(posteriors).mean();
                auto value_loss = module_output->value_loss(rewards).mean();
                auto loss = policy_loss + value_loss;

                optimizer->zero_grad();
                loss.backward();

                auto gradient_norm = rl::torchutils::compute_gradient_norm(optimizer);

                optimizer->step();

                rl::torchutils::ExecutionUnitOutput out{0, 3};
                out.scalars[0] = policy_loss;
                out.scalars[1] = value_loss;
                out.scalars[2] = gradient_norm;

                return out;
            }

        private:
            std::shared_ptr<modules::Base> module;
            std::shared_ptr<torch::optim::Optimizer> optimizer;
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_INFERENCE_UNIT_H_ */
