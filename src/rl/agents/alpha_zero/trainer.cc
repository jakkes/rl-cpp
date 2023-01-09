#include "rl/agents/alpha_zero/trainer.h"

#include <mutex>

#include <rl/agents/alpha_zero/self_play_episode.h>

#include "trainer_impl/self_play_worker.h"
#include "trainer_impl/trainer.h"


using namespace std;
using namespace trainer_impl;

namespace rl::agents::alpha_zero
{
    Trainer::Trainer(
        std::shared_ptr<rl::agents::alpha_zero::modules::Base> module,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        std::shared_ptr<rl::simulators::Base> simulator,
        const TrainerOptions &options
    ) :
        module{module}, optimizer{optimizer},
        simulator{simulator}, options{options}
    {
        
    }

    void Trainer::run(size_t duration_seconds)
    {
        auto episode_queue = make_shared<thread_safe::Queue<SelfPlayEpisode>>(1000);

        vector<unique_ptr<SelfPlayWorker>> self_play_workers{};
        self_play_workers.reserve(options.self_play_workers);
        for (int i = 0; i < options.self_play_workers; i++) {
            self_play_workers.push_back(
                make_unique<SelfPlayWorker>(
                    simulator,
                    module,
                    episode_queue,
                    SelfPlayWorkerOptions{}
                        .batchsize_(options.self_play_batchsize)
                        .discount_(options.discount)
                        .logger_(options.logger)
                        .max_episode_length_(options.max_episode_length)
                        .temperature_control_(options.self_play_temperature_control)
                        .hindsight_callback_(options.hindsight_callback)
                        .mcts_options_(
                            MCTSOptions{}
                                .dirchlet_noise_alpha_(options.self_play_dirchlet_noise_alpha)
                                .dirchlet_noise_epsilon_(options.self_play_dirchlet_noise_epsilon)
                                .discount_(options.discount)
                                .c1_(options.c1)
                                .c2_(options.c2)
                                .module_device_(options.module_device)
                                .steps_(options.self_play_mcts_steps)
                        )
                )
            );
            self_play_workers.back()->start();
        }

        
        auto optimizer_step_mtx = make_shared<mutex>();
        vector<unique_ptr<trainer_impl::Trainer>> trainers{};
        trainers.reserve(options.training_workers);

        for (int i = 0; i < options.training_workers; i++)
        {
            trainers.push_back(
                make_unique<trainer_impl::Trainer>(
                    simulator,
                    module,
                    episode_queue,
                    optimizer,
                    optimizer_step_mtx,
                    trainer_impl::TrainerOptions{}
                        .batchsize_(options.training_batchsize)
                        .gradient_norm_(options.max_gradient_norm)
                        .logger_(options.logger)
                        .min_replay_size_(options.min_replay_size)
                        .replay_size_(options.replay_size)
                        .temperature_control_(options.training_temperature_control)
                        .mcts_options_(
                            MCTSOptions{}
                                .c1_(options.c1)
                                .c2_(options.c2)
                                .dirchlet_noise_alpha_(options.training_dirchlet_noise_alpha)
                                .dirchlet_noise_epsilon_(options.training_dirchlet_noise_epsilon)
                                .discount_(options.discount)
                                .module_device_(options.module_device)
                                .steps_(options.training_mcts_steps)
                        )
                )
            );
            trainers.back()->start();
        }

        auto end_time = std::chrono::high_resolution_clock::now() + std::chrono::seconds(duration_seconds);
        while (std::chrono::high_resolution_clock::now() < end_time) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }

        for (auto &trainer_worker : trainers) {
            trainer_worker->stop();
        }
        for (auto &self_play_worker : self_play_workers) {
            self_play_worker->stop();
        }
    }
}
