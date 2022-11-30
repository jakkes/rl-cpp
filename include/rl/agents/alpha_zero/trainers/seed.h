#ifndef RL_AGENTS_ALPHA_ZERO_TRAINERS_SEED_H_
#define RL_AGENTS_ALPHA_ZERO_TRAINERS_SEED_H_

#include <memory>

#include <torch/torch.h>

#include <rl/option.h>
#include <rl/logging/client/base.h>
#include <rl/agents/alpha_zero/modules/base.h>
#include <rl/simulators/base.h>


namespace rl::agents::alpha_zero::trainers
{
    struct SEEDOptions
    {
        RL_OPTION(int, inference_batchsize) = 128;
        RL_OPTION(int, inference_max_delay_ms) = 1000;
        RL_OPTION(int, inference_mcts_steps) = 100;
        RL_OPTION(torch::Device, inference_device) = torch::kCPU;

        RL_OPTION(std::shared_ptr<rl::logging::client::Base>, logger) = nullptr;
    };

    class SEED
    {
        public:
            SEED(
                std::shared_ptr<rl::agents::alpha_zero::modules::Base> module,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                std::shared_ptr<rl::simulators::Base> simulator,
                const SEEDOptions &options={}
            );

            void run(size_t duration_seconds);

        private:
            const SEEDOptions options;
            std::shared_ptr<rl::agents::alpha_zero::modules::Base> module;
            std::shared_ptr<torch::optim::Optimizer> optimizer;
            std::shared_ptr<rl::simulators::Base> simulator;
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_TRAINERS_SEED_H_ */
