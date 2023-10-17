#ifndef RL_AGENTS_DQN_TRAINERS_APEX_IMPL_WORKER_H_
#define RL_AGENTS_DQN_TRAINERS_APEX_IMPL_WORKER_H_


#include <thread>
#include <atomic>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include <rl/agents/dqn/trainers/apex.h>
#include <rl/utils/reward/n_step_collector.h>
#include <rl/buffers/tensor.h>
#include <rl/env/base.h>


using namespace rl::agents::dqn::trainers;

namespace rl::agents::dqn::trainers::apex_impl
{
    class Worker
    {
        public:
            Worker(
                std::shared_ptr<rl::agents::dqn::Module> module,
                std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser,
                std::shared_ptr<rl::agents::dqn::policies::Base> policy,
                std::shared_ptr<rl::env::Factory> env_factory,
                std::shared_ptr<rl::buffers::Tensor> replay_buffer,
                const ApexOptions &options
            );

            void start();
            void stop();

        private:
            const ApexOptions options;
            std::shared_ptr<rl::agents::dqn::Module> module;
            std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser;
            std::shared_ptr<rl::agents::dqn::policies::Base> policy;
            std::shared_ptr<rl::env::Factory> env_factory;
            std::shared_ptr<rl::buffers::Tensor> replay_buffer;
            std::shared_ptr<rl::buffers::Tensor> local_buffer;

            std::atomic<bool> running{false};
            std::thread working_thread;

            std::vector<std::shared_ptr<rl::env::Base>> envs;
            std::vector<uint8_t> is_start_state;
            std::vector<rl::agents::dqn::utils::HindsightReplayEpisode> episodes;
            std::vector<rl::utils::reward::NStepCollector> n_step_collectors;
            std::vector<std::shared_ptr<rl::env::State>> states;

        private:
            void worker();
            void step();
            void parse_transition(const rl::utils::reward::NStepCollectorTransition &transition);
    };
}

#endif /* RL_AGENTS_DQN_TRAINERS_APEX_IMPL_WORKER_H_ */
