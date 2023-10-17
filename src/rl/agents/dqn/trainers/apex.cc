#include "rl/agents/dqn/trainers/apex.h"

#include <mutex>

#include <rl/buffers/tensor.h>
#include <rl/buffers/samplers/uniform.h>
#include <rl/cpputils/logger.h>

#include "apex_impl/trainer.h"
#include "apex_impl/worker.h"
#include "apex_impl/helpers.h"


namespace rl::agents::dqn::trainers
{
    Apex::Apex(
        std::shared_ptr<rl::agents::dqn::Module> module,
        std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        std::shared_ptr<rl::agents::dqn::policies::Base> policy,
        std::shared_ptr<rl::env::Factory> env_factory,
        const ApexOptions &options
    ) : 
        module{module}, value_parser{value_parser}, optimizer{optimizer},
        policy{policy}, env_factory{env_factory}, options{options}
    {}

    void Apex::run(int64_t duration_seconds)
    {
        auto replay = apex_impl::create_buffer(
            options.training_buffer_size,
            env_factory,
            options
        );
        
        std::vector<std::shared_ptr<apex_impl::Worker>> workers{};
        workers.reserve(options.workers);
        for (int i = 0; i < options.workers; i++) {
            workers.push_back(
                std::make_shared<apex_impl::Worker>(module, value_parser, policy, env_factory, replay, options)
            );
        }

        apex_impl::Trainer trainer{module, value_parser, optimizer, replay, options};

        trainer.start();
        for (auto &worker : workers) {
            worker->start();
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        auto end_time = start_time + std::chrono::seconds(duration_seconds);
        auto running = [&end_time] () {
            return std::chrono::high_resolution_clock::now() < end_time;
        };

        while (running()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        trainer.stop();
        for (auto &worker : workers) {
            worker->stop();
        }
    }
}
