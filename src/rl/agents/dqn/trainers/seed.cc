#include "rl/agents/dqn/trainers/seed.h"

#include <rl/buffers/tensor.h>
#include <rl/buffers/samplers/uniform.h>

#include "seed_impl/env_thread.h"
#include "seed_impl/inferer.h"
#include "seed_impl/trainer.h"
#include "seed_impl/transition_collector.h"
#include "seed_impl/helpers.h"


using namespace seed_impl;

namespace rl::agents::dqn::trainers
{

    SEED::SEED(
        std::shared_ptr<rl::agents::dqn::modules::Base> module,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        std::shared_ptr<rl::agents::dqn::policies::Base> policy,
        std::shared_ptr<rl::env::Factory> env_factory,
        const SEEDOptions &options
    ) : options{options}
    {
        this->module = module;
        this->target_module = module->clone();
        this->optimizer = optimizer;
        this->policy = policy;
        this->env_factory = env_factory;
    }

    void SEED::run(int64_t duration_seconds)
    {
        auto inferer = std::make_shared<Inferer>(module, policy, options);
        auto replay_buffer = create_buffer(options.training_buffer_size, env_factory, options);
        auto sampler = std::make_shared<rl::buffers::samplers::Uniform<rl::buffers::Tensor>>(replay_buffer);
        auto transition_queue = std::make_shared<thread_safe::Queue<rl::utils::NStepCollectorTransition>>(options.inference_replay_size);
        auto transition_collector = std::make_shared
    }
}
