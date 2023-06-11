#include "self_play_worker.h"

#include <rl/utils/reward/backpropagate.h>
#include <rl/torchutils/repeat.h>

#include "helpers.h"



using namespace torch::indexing;

namespace
{
    class InferenceUnit : public rl::torchutils::ExecutionUnit
    {
        public:
            InferenceUnit(
                int max_batchsize,
                torch::Device device,
                std::shared_ptr<modules::Base> module,
                bool use_cuda_graph
            ) : rl::torchutils::ExecutionUnit(max_batchsize, device, use_cuda_graph), module{module}
            {}
        
        private:
            std::shared_ptr<modules::Base> module;

        private:
            rl::torchutils::ExecutionUnitOutput forward(const std::vector<torch::Tensor> &inputs)
            {
                torch::NoGradGuard no_grad_guard{};
                auto module_output = module->forward(inputs[0]);
                auto policy_output = module_output->policy().get_probabilities();
                auto value_output = module_output->value_estimates();

                rl::torchutils::ExecutionUnitOutput out{2, 0};
                out.tensors[0] = policy_output;
                out.tensors[1] = value_output;

                return out;
            }
    };
}

namespace trainer_impl
{

    SelfPlayWorker::SelfPlayWorker(
        std::shared_ptr<rl::simulators::Base> simulator,
        std::shared_ptr<modules::Base> module,
        std::shared_ptr<thread_safe::Queue<SelfPlayEpisode>> episode_queue,
        std::shared_ptr<ResultTracker> result_tracker,
        const SelfPlayWorkerOptions &options
    ) : 
        simulator{simulator},
        module{module},
        episode_queue{episode_queue},
        result_tracker{result_tracker},
        options{options}
    {
        setup_inference_unit();
    }

    void SelfPlayWorker::start()
    {
        running = true;
        working_thread = std::thread(&SelfPlayWorker::worker, this);
    }

    void SelfPlayWorker::stop()
    {
        running = false;
        if (working_thread.joinable()) {
            working_thread.join();
        }
    }

    void SelfPlayWorker::set_initial_state()
    {
        auto initial_states = simulator->reset(options.batchsize);

        mcts_executor = std::make_unique<FastMCTSExecutor>(
            initial_states.states,
            get_mask(*initial_states.action_constraints),
            inference_fn_var,
            simulator,
            options.mcts_options
        );
    }

    void SelfPlayWorker::step()
    {
        mcts_executor->run();
        auto visit_counts = mcts_executor->current_visit_counts();
        auto temperature = options.temperature_control->get();
        auto probabilities = (visit_counts / std::get<0>(visit_counts.max(1, true))).pow(temperature);
        
        rl::policies::Categorical policy{probabilities};
        auto actions = policy.sample();
        mcts_executor->step(actions);
    }

    void SelfPlayWorker::worker()
    {
        torch::MultiStreamGuard stream_guard{get_cuda_streams()};

        while (running)
        {
            set_initial_state();
            while (!mcts_executor->all_terminals()) {
                step();
            }
            process_episodes();
        }
    }

    void SelfPlayWorker::process_episodes()
    {
        auto episodes = mcts_executor->get_episodes();
        auto batchsize = episodes.states.size(0);
        auto G = rl::utils::reward::backpropagate(episodes.rewards, options.discount);

        for (int i = 0; i < batchsize; i++)
        {
            SelfPlayEpisode out_episode{};
            auto length = episodes.lengths.index({i}).item().toLong();

            out_episode.states = episodes.states.index({i, Slice(None, length)});
            out_episode.masks = episodes.masks.index({i, Slice(None, length)});
            out_episode.actions = episodes.actions.index({i, Slice(None, length)});
            out_episode.collected_rewards = G.index({i, Slice(None, length)});

            enqueue_episode(out_episode);
            process_hindsight_callback(out_episode);
        }

        if (options.logger) {
            options.logger->log_scalar("AlphaZero/Reward", episodes.rewards.sum(1).mean().item().toFloat());
            options.logger->log_frequency("AlphaZero/Episode frequency", batchsize);
            result_tracker->report(episodes.lengths, episodes.actions, episodes.rewards);

            auto batchvec = torch::arange(batchsize);
            auto start_values = (
                inference_fn(
                    episodes.states.index({batchvec, 0}).to(options.module_device)
                )
                .values
            );
            auto end_values = (
                inference_fn(
                    episodes.states.index({batchvec, episodes.lengths - 1}
                )
                .to(options.module_device))
                .values
            );

            options.logger->log_scalar("AlphaZero/Start value", start_values.mean().item().toFloat());
            options.logger->log_scalar("AlphaZero/End value", end_values.mean().item().toFloat());
        }
    }

    void SelfPlayWorker::process_hindsight_callback(const SelfPlayEpisode &episode)
    {
        if (!options.hindsight_callback) {
            return;
        }

        SelfPlayEpisode hindsight_episode{};
        hindsight_episode.states = episode.states.clone();
        hindsight_episode.masks = episode.masks.clone();
        hindsight_episode.collected_rewards = episode.collected_rewards.clone();
        auto should_enqueue = options.hindsight_callback(&hindsight_episode);

        if (should_enqueue) {
            enqueue_episode(hindsight_episode);

            if (options.logger) {
                options.logger->log_scalar(
                    "AlphaZero/Hindsight reward",
                    hindsight_episode.collected_rewards.index({0}).item().toFloat()
                );
                options.logger->log_frequency(
                    "AlphaZero/Hindsight episode frequency", 1
                );
            }
        }
    }

    void SelfPlayWorker::enqueue_episode(const SelfPlayEpisode &episode)
    {
        bool enqueued{false};
        while (!enqueued) {
            enqueued = episode_queue->enqueue(episode, std::chrono::seconds(5));
        }
    }

    void SelfPlayWorker::setup_inference_unit()
    {
        inference_unit = std::make_unique<InferenceUnit>(
            options.batchsize,
            options.module_device,
            module,
            options.enable_cuda_graph_inference
        );
        inference_unit->operator()({simulator->reset(options.batchsize).states.to(options.module_device)});
    }

    FastMCTSInferenceResult SelfPlayWorker::inference_fn(const torch::Tensor &states) {
        auto outputs = inference_unit->operator()({states});
        return FastMCTSInferenceResult{
            outputs.tensors[0],
            outputs.tensors[1]
        };
    }
}
