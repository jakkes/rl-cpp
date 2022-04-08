#include "rl/agents/ppo/trainers/seed.h"


namespace rl::agents::ppo::trainers
{
    SEED::SEED(
        std::shared_ptr<rl::agents::ppo::Module> model,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        std::shared_ptr<rl::env::Factory> env_factory,
        const SEEDOptions &options={}
    ) :
    model{model}, optimizer{optimizer}, {env_factory}, options{options}
    {}

    template<class Rep, class Period>
    SEED::run(std::chrono::duration<Rep, Period> duration)
    {
        auto start = std::chrono::steady_clock::now();
        auto end = start + duration;
    }
}
