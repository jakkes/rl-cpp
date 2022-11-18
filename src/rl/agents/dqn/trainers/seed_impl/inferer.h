#ifndef RL_AGENTS_DQN_TRAINERS_SEED_IMPL_INFERENCE_H_
#define RL_AGENTS_DQN_TRAINERS_SEED_IMPL_INFERENCE_H_

#include <memory>
#include <atomic>
#include <mutex>
#include <vector>
#include <thread>
#include <functional>
#include <deque>

#include <torch/torch.h>
#include <thread_safe/collections/queue.h>

#include <rl/agents/dqn/trainers/seed.h>
#include <rl/agents/dqn/modules/base.h>
#include <rl/agents/dqn/policies/base.h>
#include <rl/policies/categorical.h>


namespace seed_impl
{

    struct InferenceResult
    {
        torch::Tensor action;
        torch::Tensor value;
        torch::Tensor advantage;
    };

    class InferenceBatch
    {
        public:
            InferenceBatch(
                std::shared_ptr<rl::agents::dqn::modules::Base> module,
                std::shared_ptr<rl::agents::dqn::policies::Base> policy,
                std::function<void(InferenceBatch*)> stale_callback,
                const rl::agents::dqn::trainers::SEEDOptions *options
            );

            ~InferenceBatch();

            inline
            bool executed() const { return executed_; }

            inline
            bool stale() const { return stale_; }

            std::unique_ptr<InferenceResult> get(int64_t id);

            int64_t try_add(const torch::Tensor &state, const torch::Tensor &mask);
        
            void execute();

        private:
            std::shared_ptr<rl::agents::dqn::modules::Base> module;
            std::shared_ptr<rl::agents::dqn::policies::Base> policy;
            std::function<void(InferenceBatch*)> stale_callback;
            const rl::agents::dqn::trainers::SEEDOptions *options;

            std::mutex add_mtx{};
            std::condition_variable add_cv{};
            std::atomic<bool> stale_{false};

            std::condition_variable executed_cv{};
            std::atomic<bool> executed_{false};
            
            std::thread worker_thread;

            std::vector<torch::Tensor> states{};
            std::vector<torch::Tensor> masks{};
            torch::Tensor value;
            torch::Tensor actions;
            torch::Tensor advantage;

            std::chrono::_V2::system_clock::time_point start_time;

        private:
            inline bool full() { return size() >= options->inference_batchsize; }
            inline bool empty() { return size() == 0; }
            inline size_t size() { return states.size(); }
            void start_worker();
            void worker();
    };

    class InferenceResultFuture
    {
        public:
            InferenceResultFuture(
                std::shared_ptr<InferenceBatch> batch,
                int64_t id
            ) : batch{batch}, id{id}
            {}

            inline
            bool ready() const { return batch->executed(); }

            inline
            std::unique_ptr<InferenceResult> result() { return batch->get(id); }

        private:
            const std::shared_ptr<InferenceBatch> batch;
            const int64_t id;
    };

    class Inferer
    {
        public:
            Inferer(
                std::shared_ptr<rl::agents::dqn::modules::Base> module,
                std::shared_ptr<rl::agents::dqn::policies::Base> policy,
                const rl::agents::dqn::trainers::SEEDOptions &options
            );

            std::unique_ptr<InferenceResultFuture> infer(
                const torch::Tensor &state,
                const torch::Tensor &mask
            );

            void start();
            void stop();
        
        private:
            std::shared_ptr<rl::agents::dqn::modules::Base> module;
            std::shared_ptr<rl::agents::dqn::policies::Base> policy;
            const rl::agents::dqn::trainers::SEEDOptions options;

            std::atomic<bool> running{false};
            std::thread worker_thread;

            thread_safe::Queue<std::shared_ptr<InferenceBatch>> batch_queue{4};

            std::mutex infer_mtx{};
            std::deque<std::shared_ptr<InferenceBatch>> batches{};
        
        private:
            void worker();
            void batch_stale_callback(InferenceBatch *batch);
            void new_batch();
    };
}

#endif /* RL_AGENTS_DQN_TRAINERS_SEED_IMPL_INFERENCE_H_ */
