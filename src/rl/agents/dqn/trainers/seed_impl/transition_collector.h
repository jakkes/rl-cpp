#ifndef RL_AGENTS_DQN_TRAINERS_SEED_IMPL_TRANSITION_COLLECTOR_H_
#define RL_AGENTS_DQN_TRAINERS_SEED_IMPL_TRANSITION_COLLECTOR_H_


#include <memory>
#include <thread>
#include <atomic>

#include <thread_safe/collections/queue.h>

#include <rl/utils/reward/n_step_collector.h>
#include <rl/buffers/tensor.h>
#include <rl/agents/dqn/trainers/seed.h>


namespace seed_impl
{
    class TransitionCollector
    {
        public:
            TransitionCollector(
                std::shared_ptr<thread_safe::Queue<rl::utils::reward::NStepCollectorTransition>> transition_queue,
                std::shared_ptr<rl::buffers::Tensor> training_buffer,
                std::shared_ptr<rl::env::Factory> env_factory,
                const rl::agents::dqn::trainers::SEEDOptions &options
            );

            void start();
            void stop();
        
        private:
            const rl::agents::dqn::trainers::SEEDOptions options;
            std::shared_ptr<rl::buffers::Tensor> training_buffer;
            std::shared_ptr<rl::buffers::Tensor> inference_buffer;
            std::shared_ptr<thread_safe::Queue<rl::utils::reward::NStepCollectorTransition>> transition_queue;

            std::atomic<bool> running{false};
            std::thread working_thread;

        private:
            void worker();
    };
}

#endif /* RL_AGENTS_DQN_TRAINERS_SEED_IMPL_TRANSITION_COLLECTOR_H_ */
