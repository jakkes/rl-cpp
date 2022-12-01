#ifndef RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_SELF_PLAY_WORKER_H_
#define RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_SELF_PLAY_WORKER_H_


#include <atomic>
#include <thread>

#include <torch/torch.h>

#include <rl/option.h>
#include <rl/simulators/base.h>
#include <rl/agents/alpha_zero/alpha_zero.h>

using namespace rl::agents::alpha_zero;

namespace trainer_impl
{
    struct SelfPlayWorkerOptions
    {

    };

    class SelfPlayWorker
    {
        public:
            SelfPlayWorker(
                std::shared_ptr<rl::simulators::Base> simulator,
                std::shared_ptr<modules::Base> module,
                const SelfPlayWorkerOptions &options={}
            );

            void start();
            void stop();

        private:
            std::shared_ptr<rl::simulators::Base> simulator;
            std::shared_ptr<modules::Base> module;
            const SelfPlayWorkerOptions options;

            std::atomic<bool> running{false};
            std::thread working_thread;        

        private:
            void worker();
            void step();
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_TRAINER_IMPL_SELF_PLAY_WORKER_H_ */
