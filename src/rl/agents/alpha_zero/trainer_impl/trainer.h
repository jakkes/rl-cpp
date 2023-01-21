#ifndef RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_TRAINER_H_
#define RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_TRAINER_H_


#include <atomic>
#include <mutex>
#include <thread>

#include <thread_safe/collections/queue.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>

#include <rl/option.h>
#include <rl/logging/client/base.h>
#include <rl/utils/float_control/fixed.h>
#include <rl/agents/alpha_zero/alpha_zero.h>
#include <rl/buffers/buffers.h>


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

        RL_OPTION(std::shared_ptr<rl::logging::client::Base>, logger) = nullptr;
    };

    class Trainer
    {
        public:
            Trainer(
                std::shared_ptr<rl::simulators::Base> simulator,
                std::shared_ptr<modules::Base> module,
                std::shared_ptr<thread_safe::Queue<SelfPlayEpisode>> episode_queue,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                std::shared_ptr<std::mutex> optimizer_step_mtx,
                const TrainerOptions &options={}
            );

            void start();
            void stop();
        
        private:
            std::shared_ptr<rl::simulators::Base> simulator;
            std::shared_ptr<modules::Base> module;
            std::shared_ptr<thread_safe::Queue<SelfPlayEpisode>> episode_queue;
            std::shared_ptr<torch::optim::Optimizer> optimizer;
            std::shared_ptr<std::mutex> optimizer_step_mtx;
            const TrainerOptions options;

            std::shared_ptr<rl::buffers::Tensor> buffer;
            std::unique_ptr<rl::buffers::samplers::Uniform<rl::buffers::Tensor>> sampler;

            c10::cuda::CUDAStream cuda_stream{c10::cuda::getStreamFromPool()};
            std::unique_ptr<at::cuda::CUDAGraph> inference_graph = nullptr;
            torch::Tensor inference_input, inference_policy_output, inference_value_output;

            std::atomic<bool> running{false};
            std::thread working_thread;
            std::thread queue_consuming_thread;

            std::function<MCTSInferenceResult(const torch::Tensor &)> inference_fn;
        
        private:
            void init_buffer();
            void worker();
            void queue_consumer();
            void step();
            torch::Tensor get_target_policy(const torch::Tensor &states, const torch::Tensor &masks);

            MCTSInferenceResult cuda_graph_inference_fn(const torch::Tensor &states);
            void cuda_graph_inference_setup();

            inline
            MCTSInferenceResult cpu_inference_fn(const torch::Tensor &states) {
                torch::InferenceMode guard{};
                auto module_output = module->forward(states);
                return MCTSInferenceResult{module_output->policy(), module_output->value_estimates()};
            }

            void inference_fn_setup();
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_TRAINER_H_ */
