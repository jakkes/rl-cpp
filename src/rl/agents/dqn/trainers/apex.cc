#include "rl/agents/dqn/trainers/apex.h"

#include <mutex>

#include <rl/buffers/tensor.h>
#include <rl/buffers/samplers/uniform.h>
#include <rl/cpputils/logger.h>

#include "apex_impl/trainer.h"
#include "apex_impl/worker.h"
#include "apex_impl/helpers.h"
#include "apex_impl/execution_units.h"


namespace rl::agents::dqn::trainers
{
    static
    std::shared_ptr<apex_impl::InferenceUnit> get_initialized_inference_unit(
        std::shared_ptr<rl::agents::dqn::Module> module,
        std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser,
        std::shared_ptr<rl::env::Factory> env_factory,
        const ApexOptions &options
    ) {
        auto inference_unit = std::make_shared<apex_impl::InferenceUnit>(
            module, value_parser, options
        );

        auto env = env_factory->get();
        auto state = env->reset();
        inference_unit->operator()({
            state->state.to(options.network_device).unsqueeze(0),
            apex_impl::get_mask(*state->action_constraint).to(options.network_device).unsqueeze(0)
        });

        return inference_unit;
    }

    static
    std::shared_ptr<apex_impl::TrainingUnit> get_initialized_training_unit(
        std::shared_ptr<rl::agents::dqn::Module> module,
        std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        std::shared_ptr<rl::env::Factory> env_factory,
        const ApexOptions &options
    ) {
        auto training_unit = std::make_shared<apex_impl::TrainingUnit>(
            module, value_parser, optimizer, options
        );

        auto state = env_factory->get()->reset();

        training_unit->initialize_graph(*state);
        return training_unit;
    }

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
        
        auto inference_unit = get_initialized_inference_unit(
            module, value_parser, env_factory, options
        );

        auto training_unit = get_initialized_training_unit(
            module, value_parser, optimizer, env_factory, options
        );

        std::vector<std::shared_ptr<apex_impl::Worker>> workers{};
        workers.reserve(options.workers);
        for (int i = 0; i < options.workers; i++) {
            workers.push_back(
                std::make_shared<apex_impl::Worker>(
                    inference_unit, policy, env_factory, replay, options
                )
            );
        }

        apex_impl::Trainer trainer{training_unit, replay, options};

        trainer.start();
        for (auto &worker : workers) {
            worker->start();
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        auto end_time = start_time + std::chrono::seconds(duration_seconds);
        auto next_checkpoint = start_time + std::chrono::seconds(options.checkpoint_callback_period_seconds);
        size_t checkpoint_version = 0;

        auto running = [&end_time] () {
            return std::chrono::high_resolution_clock::now() < end_time;
        };

        while (running()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));

            if (options.checkpoint_callback && std::chrono::high_resolution_clock::now() >= next_checkpoint) {
                checkpoint_version++;
                options.checkpoint_callback(
                    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count(),
                    checkpoint_version
                );
                next_checkpoint = std::chrono::high_resolution_clock::now() + std::chrono::seconds(options.checkpoint_callback_period_seconds);
            }

            if (options.logger) {
                options.logger->log_scalar("ApexDQN/Buffer size", replay->size());
            }
        }

        trainer.stop();
        for (auto &worker : workers) {
            worker->stop();
        }
    }
}
