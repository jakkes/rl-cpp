#include "rl/agents/dqn/trainers/seed.h"


namespace rl::agents::dqn::trainers
{
    SEED::SEED(
        std::shared_ptr<rl::agents::dqn::modules::Base> module,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        std::shared_ptr<rl::env::Factory> env_factory,
        const SEEDOptions &options
    ) : options{options}
    {
        this->module = module;
        this->target_module = module->clone();
        this->optimizer = optimizer;
        this->env_factory = env_factory;
    }
}
