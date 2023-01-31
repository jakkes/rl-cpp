#include "rl/agents/alpha_zero/trainer.h"

#include <mutex>

#include <rl/agents/alpha_zero/self_play_episode.h>

#include "trainer_impl/self_play_worker.h"
#include "trainer_impl/trainer.h"
#include "trainer_impl/helpers.h"


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
        init_buffer();
    }

    void Trainer::init_buffer()
    {
        auto states = simulator->reset(1);
        auto state = states.states.squeeze(0);
        auto mask = get_mask(*states.action_constraints).squeeze(0);
        
        buffer = std::make_shared<rl::buffers::Tensor>(
            options.replay_size,
            std::vector{
                state.sizes().vec(),
                mask.sizes().vec(),
                std::vector<int64_t>{}
            },
            std::vector{
                torch::TensorOptions{}.dtype(state.dtype()).device(options.replay_device),
                torch::TensorOptions{}.dtype(mask.dtype()).device(options.replay_device),
                torch::TensorOptions{}.dtype(torch::kFloat32).device(options.replay_device)
            }
        );

        sampler = std::make_shared<rl::buffers::samplers::Uniform<rl::buffers::Tensor>>(buffer);
    }

    void Trainer::run(size_t duration_seconds)
    {
        running = true;
        episode_queue = make_shared<thread_safe::Queue<SelfPlayEpisode>>(1000);

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
                        .module_device_(options.module_device)
                        .enable_cuda_graph_inference_(options.enable_inference_cuda_graph)
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
                                .sim_device_(options.sim_device)
                                .steps_(options.self_play_mcts_steps)
                        )
                )
            );
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
                    sampler,
                    optimizer,
                    optimizer_step_mtx,
                    trainer_impl::TrainerOptions{}
                        .batchsize_(options.training_batchsize)
                        .logger_(options.logger)
                        .min_replay_size_(options.min_replay_size)
                        .module_device_(options.module_device)
                        .replay_size_(options.replay_size)
                        .enable_cuda_graph_training_(options.enable_training_cuda_graph)
                        .enable_cuda_graph_inference_(options.enable_inference_cuda_graph)
                        .temperature_control_(options.training_temperature_control)
                        .mcts_options_(
                            MCTSOptions{}
                                .c1_(options.c1)
                                .c2_(options.c2)
                                .dirchlet_noise_alpha_(options.training_dirchlet_noise_alpha)
                                .dirchlet_noise_epsilon_(options.training_dirchlet_noise_epsilon)
                                .discount_(options.discount)
                                .module_device_(options.module_device)
                                .sim_device_(options.sim_device)
                                .steps_(options.training_mcts_steps)
                        )
                )
            );
        }

        queue_consuming_thread = std::thread(&Trainer::queue_consumer, this);

        for (auto &worker : self_play_workers) {
            worker->start();
        }

        for (auto &trainer : trainers) {
            trainer->start();
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
        if (queue_consuming_thread.joinable()) {
            queue_consuming_thread.join();
        }
    }

    void Trainer::queue_consumer()
    {
        while (running)
        {
            auto episode_ptr = episode_queue->dequeue(std::chrono::seconds(5));
            if (!episode_ptr) {
                continue;
            }

            auto episode = *episode_ptr;
            buffer->add({episode.states.to(options.replay_device), episode.masks.to(options.replay_device), episode.collected_rewards.to(options.replay_device)});
        }
    }
}
