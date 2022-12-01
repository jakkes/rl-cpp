#ifndef RL_AGENTS_ALPHA_ZERO_TRAINER_H_
#define RL_AGENTS_ALPHA_ZERO_TRAINER_H_


#include <rl/option.h>

namespace rl::agents::alpha_zero
{
    struct TrainerOptions
    {
        RL_OPTION(int, self_play_batchsize) = 32;
        RL_OPTION(int, self_play_workers) = 100;
        RL_OPTION(torch::Device, module_device) = torch::kCPU;

        RL_OPTION(std::shared_ptr<rl::logging::client::Base>, logger) = nullptr;
    };

    class Trainer
    {
        public:
            Trainer(
                std::shared_ptr<rl::agents::alpha_zero::modules::Base> module,
                std::shared_ptr<torch::optim::Optimizer> optimizer,
                std::shared_ptr<rl::simulators::Base> simulator,
                const TrainerOptions &options={}
            );

            void run(size_t duration_seconds);

        private:
            const TrainerOptions options;
            std::shared_ptr<rl::agents::alpha_zero::modules::Base> module;
            std::shared_ptr<torch::optim::Optimizer> optimizer;
            std::shared_ptr<rl::simulators::Base> simulator;
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_TRAINER_H_ */
