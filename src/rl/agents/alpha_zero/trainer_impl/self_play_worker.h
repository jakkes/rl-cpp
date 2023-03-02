#ifndef RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_SELF_PLAY_WORKER_H_
#define RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_SELF_PLAY_WORKER_H_


#include <atomic>
#include <thread>
#include <vector>
#include <functional>
#include <memory>

#include <thread_safe/collections/queue.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>

#include <rl/option.h>
#include <rl/simulators/base.h>
#include <rl/logging/client/base.h>
#include <rl/agents/alpha_zero/alpha_zero.h>
#include <rl/torchutils/execution_unit.h>

#include "result_tracker.h"


using namespace rl::agents::alpha_zero;

namespace trainer_impl
{
    struct SelfPlayWorkerOptions
    {
        RL_OPTION(int, batchsize) = 32;
        RL_OPTION(int, max_episode_length) = 100;
        RL_OPTION(std::shared_ptr<rl::utils::float_control::Base>, temperature_control) = std::make_shared<rl::utils::float_control::Fixed>(1.0f);
        RL_OPTION(float, discount) = 1.0f;
        RL_OPTION(FastMCTSExecutorOptions, mcts_options) = FastMCTSExecutorOptions{};

        RL_OPTION(torch::Device, module_device) = torch::kCPU;
        RL_OPTION(torch::Device, sim_device) = torch::kCPU;
        RL_OPTION(bool, enable_cuda_graph_inference) = true;

        RL_OPTION(std::shared_ptr<rl::logging::client::Base>, logger) = nullptr;
        RL_OPTION(std::function<bool(SelfPlayEpisode*)>, hindsight_callback) = nullptr;
    };

    class SelfPlayWorker
    {
        public:
            SelfPlayWorker() = default;
            SelfPlayWorker(
                std::shared_ptr<rl::simulators::Base> simulator,
                std::shared_ptr<modules::Base> module,
                std::shared_ptr<thread_safe::Queue<SelfPlayEpisode>> episode_queue,
                std::shared_ptr<ResultTracker> result_tracker,
                const SelfPlayWorkerOptions &options={}
            );

            void start();
            void stop();

        private:
            std::shared_ptr<rl::simulators::Base> simulator;
            std::shared_ptr<modules::Base> module;
            std::shared_ptr<thread_safe::Queue<SelfPlayEpisode>> episode_queue;
            std::shared_ptr<ResultTracker> result_tracker;
            const SelfPlayWorkerOptions options;

            std::unique_ptr<rl::torchutils::ExecutionUnit> inference_unit;
            std::function<MCTSInferenceResult(const torch::Tensor &)> inference_fn_var = std::bind(&SelfPlayWorker::inference_fn, this, std::placeholders::_1);

            std::atomic<bool> running{false};
            std::thread working_thread;

            std::unique_ptr<FastMCTSExecutor> mcts_executor;

        private:
            void worker();
            void step();
            void set_initial_state();
            void process_episodes();
            void enqueue_episode(const SelfPlayEpisode &episode);

            void setup_inference_unit();
            MCTSInferenceResult inference_fn(const torch::Tensor &states);
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_SELF_PLAY_WORKER_H_ */
