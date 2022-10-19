#include "rl/agents/dqn/trainers/seed.h"

#include <rl/buffers/tensor.h>
#include <rl/buffers/samplers/uniform.h>
#include <rl/cpputils/logger.h>

#include "seed_impl/env_thread.h"
#include "seed_impl/inferer.h"
#include "seed_impl/trainer.h"
#include "seed_impl/transition_collector.h"
#include "seed_impl/helpers.h"

using namespace seed_impl;

namespace rl::agents::dqn::trainers
{
    static
    auto LOGGER = rl::cpputils::get_logger("SEEDDQN");


    SEED::SEED(
        std::shared_ptr<rl::agents::dqn::modules::Base> module,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        std::shared_ptr<rl::agents::dqn::policies::Base> policy,
        std::shared_ptr<rl::env::Factory> env_factory,
        const SEEDOptions &options) : options{options}
    {
        this->module = module;
        this->optimizer = optimizer;
        this->policy = policy;
        this->env_factory = env_factory;
    }

    void SEED::run(int64_t duration_seconds)
    {
        auto inferer = std::make_shared<Inferer>(
            module, 
            policy,
            options
        );
        auto replay_buffer = create_buffer(
            options.training_buffer_size,
            env_factory,
            options
        );
        auto sampler = std::make_shared<rl::buffers::samplers::Uniform<rl::buffers::Tensor>>(replay_buffer);
        auto transition_queue = std::make_shared<thread_safe::Queue<rl::utils::NStepCollectorTransition>>(options.inference_replay_size);
        auto transition_collector = std::make_shared<TransitionCollector>(
            transition_queue,
            replay_buffer,
            env_factory,
            options
        );

        std::vector<std::unique_ptr<EnvThread>> env_threads{};
        env_threads.reserve(options.env_workers);
        for (int i = 0; i < options.env_workers; i++) {
            env_threads.emplace_back(new EnvThread{env_factory, inferer, transition_queue, options});
        }

        auto trainer = std::make_shared<Trainer>(module, optimizer, env_factory, sampler, options);

        auto start_time = std::chrono::high_resolution_clock::now();
        auto end_time = start_time + std::chrono::seconds(duration_seconds);
        auto running = [&end_time] () {
            return std::chrono::high_resolution_clock::now() < end_time;
        };

        transition_collector->start();
        for (auto &env_thread : env_threads) {
            env_thread->start();
        }

        while (running() && replay_buffer->size() < options.minimum_replay_buffer_size) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        LOGGER->info("Starting trainer.");
        trainer->start();

        while (running()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        for (auto &env_thread : env_threads) {
            env_thread->stop();
        }
        transition_collector->stop();
        trainer->stop();
    }
}
