#include "rl/agents/alpha_zero/trainer.h"

#include "trainer_impl/self_play_worker.h"
#include "trainer_impl/trainer.h"


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
        auto episode_queue = std::make_shared<thread_safe::Queue<trainer_impl::SelfPlayEpisode>>(1000);

        std::vector<std::unique_ptr<trainer_impl::SelfPlayWorker>> self_play_workers{};
        self_play_workers.reserve(options.self_play_workers);
        for (int i = 0; i < options.self_play_workers; i++) {
            self_play_workers.push_back(
                std::make_unique<trainer_impl::SelfPlayWorker>(
                    simulator,
                    module,
                    episode_queue,
                    trainer_impl::SelfPlayWorkerOptions{}
                        .batchsize_(options.self_play_batchsize)
                        .discount_(options.discount)
                        .logger_(options.logger)
                        .max_episode_length_(options.max_episode_length)
                        .temperature_(options.self_play_temperature)
                        .mcts_options_(options.self_play_mcts_options)
                )
            );
            self_play_workers.back()->start();
        }

        trainer_impl::Trainer trainer_worker{
            simulator,
            module,
            episode_queue,
            optimizer,
            trainer_impl::TrainerOptions{}
                .batchsize_(options.training_batchsize)
                .gradient_norm_(options.max_gradient_norm)
                .logger_(options.logger)
                .mcts_options_(options.training_mcts_options)
                .min_replay_size_(options.min_replay_size)
                .replay_size_(options.replay_size)
                .temperature_(options.training_temperature)
        };

        trainer_worker.start();

        auto end_time = std::chrono::high_resolution_clock::now() + std::chrono::seconds(duration_seconds);
        while (std::chrono::high_resolution_clock::now() < end_time) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }

        trainer_worker.stop();
        for (auto &self_play_worker : self_play_workers) {
            self_play_worker->stop();
        }
    }
}
