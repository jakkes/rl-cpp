#ifndef RL_AGENTS_DQN_TRAINERS_APEX_IMPL_TRAINER_H_
#define RL_AGENTS_DQN_TRAINERS_APEX_IMPL_TRAINER_H_


#include <thread>
#include <mutex>
#include <atomic>

#include <torch/torch.h>

#include <rl/agents/dqn/trainers/apex.h>
#include <rl/buffers/tensor.h>
#include <rl/buffers/samplers/uniform.h>


using namespace rl::agents::dqn::trainers;

namespace rl::agents::dqn::trainers::apex_impl
{
    class Trainer
    {
        public:
            Trainer(
                std::shared_ptr<rl::agents::dqn::Module> module,
                std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                std::shared_ptr<rl::buffers::Tensor> replay_buffer,
                const ApexOptions &options
            );

            void start();
            void stop();

        private:
            const ApexOptions options;
            std::shared_ptr<rl::agents::dqn::Module> module;
            std::shared_ptr<rl::agents::dqn::Module> target_module;
            std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser;
            std::shared_ptr<torch::optim::Optimizer> optimizer;
            std::shared_ptr<rl::agents::dqn::policies::Base> policy;
            std::shared_ptr<rl::env::Factory> env_factory;
            std::shared_ptr<rl::buffers::samplers::Uniform<rl::buffers::Tensor>> replay_buffer;

            std::atomic<bool> running{false};
            std::thread working_thread;

        private:
            void worker();
            void step();
            void target_network_update();
    };
}

#endif /* RL_AGENTS_DQN_TRAINERS_APEX_IMPL_TRAINER_H_ */
