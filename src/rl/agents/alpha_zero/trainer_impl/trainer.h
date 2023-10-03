#ifndef RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_TRAINER_H_
#define RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_TRAINER_H_


#include <atomic>
#include <mutex>
#include <thread>

#include <torch/torch.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>

#include <rl/option.h>
#include <rl/logging/client/base.h>
#include <rl/utils/float_control/fixed.h>
#include <rl/agents/alpha_zero/alpha_zero.h>
#include <rl/buffers/buffers.h>
#include <rl/torchutils/execution_unit.h>

#include "execution_units.h"


using namespace rl::agents::alpha_zero;

namespace trainer_impl
{
    struct TrainerOptions
    {
        RL_OPTION(int, batchsize) = 128;
        RL_OPTION(int64_t, replay_size) = 1000;
        RL_OPTION(std::shared_ptr<rl::utils::float_control::Base>, temperature_control) = std::make_shared<rl::utils::float_control::Fixed>(1.0f);
        RL_OPTION(float, gradient_norm) = 40.0f;
        RL_OPTION(size_t, min_replay_size) = 1000;
        RL_OPTION(MCTSOptions, mcts_options) = MCTSOptions{};

        RL_OPTION(torch::Device, module_device) = torch::kCPU;
        RL_OPTION(bool, enable_cuda_graph_training) = true;
        RL_OPTION(bool, enable_cuda_graph_inference) = true;

        RL_OPTION(std::shared_ptr<rl::logging::client::Base>, logger) = nullptr;
    };

    class Trainer
    {
        public:
            Trainer(
                std::shared_ptr<rl::simulators::Base> simulator,
                std::shared_ptr<modules::Base> module,
                std::shared_ptr<rl::buffers::samplers::Uniform<rl::buffers::Tensor>> sampler,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                std::shared_ptr<std::mutex> optimizer_step_mtx,
                const TrainerOptions &options={}
            );

            void start();
            void stop();
        
        private:
            std::shared_ptr<rl::simulators::Base> simulator;
            std::shared_ptr<modules::Base> module;
            std::shared_ptr<torch::optim::Optimizer> optimizer;
            std::shared_ptr<std::mutex> optimizer_step_mtx;
            const TrainerOptions options;

            std::shared_ptr<rl::buffers::samplers::Uniform<rl::buffers::Tensor>> sampler;

            std::atomic<bool> running{false};
            std::thread working_thread;

            std::function<MCTSInferenceResult(const torch::Tensor &)> inference_fn_var = std::bind(&Trainer::inference_fn, this, std::placeholders::_1);
            std::unique_ptr<InferenceUnit> inference_unit;
            std::unique_ptr<TrainingUnit> training_unit;

        private:
            void init_buffer();
            void worker();
            void step();
            torch::Tensor get_target_policy(const torch::Tensor &states, const torch::Tensor &masks);

            void setup_inference_unit();
            MCTSInferenceResult inference_fn(const torch::Tensor &states);
            void setup_training_unit();
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_TRAINER_H_ */
