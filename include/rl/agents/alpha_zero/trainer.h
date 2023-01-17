#ifndef RL_AGENTS_ALPHA_ZERO_TRAINER_H_
#define RL_AGENTS_ALPHA_ZERO_TRAINER_H_


#include <memory>
#include <functional>

#include <rl/option.h>
#include <rl/utils/float_control/fixed.h>

#include "mcts.h"
#include "self_play_episode.h"


namespace rl::agents::alpha_zero
{
    struct TrainerOptions
    {
        RL_OPTION(int, max_episode_length) = 100;
        RL_OPTION(int, self_play_batchsize) = 32;
        RL_OPTION(int, self_play_workers) = 1;
        RL_OPTION(int, self_play_mcts_steps) = 100;
        RL_OPTION(float, self_play_dirchlet_noise_alpha) = 0.1f;
        RL_OPTION(float, self_play_dirchlet_noise_epsilon) = 0.5f;
        RL_OPTION(std::shared_ptr<rl::utils::float_control::Base>, self_play_temperature_control) = std::make_shared<rl::utils::float_control::Fixed>(1.0f);

        RL_OPTION(torch::Device, module_device) = torch::kCPU;

        RL_OPTION(float, discount) = 1.0f;
        RL_OPTION(float, c1) = 1.25f;
        RL_OPTION(float, c2) = 19652;

        RL_OPTION(int, training_batchsize) = 128;
        RL_OPTION(int, training_workers) = 1;
        RL_OPTION(int, replay_size) = 10000;
        RL_OPTION(int, min_replay_size) = 1000;
        RL_OPTION(int, training_mcts_steps) = 100;
        RL_OPTION(float, training_dirchlet_noise_alpha) = 0.1f;
        RL_OPTION(float, training_dirchlet_noise_epsilon) = 0.5f;
        RL_OPTION(std::shared_ptr<rl::utils::float_control::Base>, training_temperature_control) = std::make_shared<rl::utils::float_control::Fixed>(1.0f);

        RL_OPTION(std::shared_ptr<rl::logging::client::Base>, logger) = nullptr;
        RL_OPTION(std::function<bool(SelfPlayEpisode*)>, hindsight_callback) = nullptr;
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
