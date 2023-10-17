#include "worker.h"

#include <c10/cuda/CUDAStream.h>
#include <rl/cpputils/logger.h>

#include "helpers.h"


namespace rl::agents::dqn::trainers::apex_impl
{
    static
    auto LOGGER = rl::cpputils::get_logger("ApexDQN-Worker");

    Worker::Worker(
        std::shared_ptr<rl::agents::dqn::Module> module,
        std::shared_ptr<rl::agents::dqn::value_parsers::Base> value_parser,
        std::shared_ptr<rl::agents::dqn::policies::Base> policy,
        std::shared_ptr<rl::env::Factory> env_factory,
        std::shared_ptr<rl::buffers::Tensor> replay_buffer,
        const ApexOptions &options
    ) : options{options}
    {
        this->module = module;
        this->value_parser = value_parser;
        this->policy = policy;
        this->env_factory = env_factory;
        this->replay_buffer = replay_buffer;

        this->local_buffer = create_buffer(
            options.inference_replay_size,
            env_factory,
            options
        );
    }

    void Worker::start()
    {
        running = true;
        working_thread = std::thread(&Worker::worker, this);
    }

    void Worker::stop()
    {
        running = false;
        if (working_thread.joinable()) working_thread.join();
    }

    void Worker::worker()
    {
        torch::StreamGuard stream_guard{c10::cuda::getStreamFromPool()};
        torch::InferenceMode inference_guard{};

        envs.reserve(options.worker_batchsize);
        n_step_collectors.reserve(options.worker_batchsize);
        for (int i = 0; i < options.worker_batchsize; i++) {
            envs.push_back(env_factory->get());
            n_step_collectors.emplace_back(options.n_step, options.discount);
        }

        states.resize(options.worker_batchsize);
        episodes.resize(options.worker_batchsize);
        for (int i = 0; i < options.worker_batchsize; i++) {
            states[i] = envs[i]->reset();
            episodes[i].states.push_back(states[i]);
        }

        is_start_state.resize(options.worker_batchsize, 1);

        LOGGER->info("Starting worker");
        while (running) {
            step();
        }
        LOGGER->info("Stopping worker");
    }

    void Worker::step()
    {
        std::vector<torch::Tensor> states{};
        states.resize(options.worker_batchsize);
        std::vector<torch::Tensor> masks{};
        masks.resize(options.worker_batchsize);

        for (int i = 0; i < options.worker_batchsize; i++) {
            states[i] = this->states[i]->state;
            masks[i] = get_mask(*this->states[i]->action_constraint);
        }

        auto tstates = torch::stack(states, 0).to(options.network_device);
        auto tmasks = torch::stack(masks, 0).to(options.network_device);

        auto outputs = module->forward(tstates);
        auto values = value_parser->values(outputs, tmasks);

        auto policy = this->policy->policy(values);
        policy->include(std::make_shared<rl::policies::constraints::CategoricalMask>(tmasks));

        auto actions = policy->sample().to(options.environment_device);
        auto max_values = std::get<0>(values.max(1));

        for (int i = 0; i < options.worker_batchsize; i++)
        {
            if (is_start_state[i]) {
                is_start_state[i] = 0;
                if (options.logger) {
                    options.logger->log_scalar("ApexDQN/Start value", max_values.index({i}).item().toFloat());
                }
            }

            auto action = actions.index({i});

            auto observation = envs[i]->step(action);

            auto transitions = n_step_collectors[i].step(
                this->states[i],
                actions.index({i}),
                observation->reward,
                observation->terminal
            );

            for (const auto &transition : transitions) {
                parse_transition(transition);
            }

            if (observation->terminal) {
                this->states[i] = envs[i]->reset();
                this->is_start_state[i] = 1;

                if (options.logger) {
                    options.logger->log_frequency("ApexDQN/Episode rate", 1);
                    options.logger->log_scalar("ApexDQN/End value", max_values.index({i}).item().toFloat());
                }
            }
            else {
                this->states[i] = observation->state;
            }

            episodes[i].rewards.push_back(observation->reward);
            episodes[i].actions.push_back(action);
            episodes[i].states.push_back(observation->state);

            if (observation->terminal) {
                if (options.hindsight_replay_callback) {
                    auto add_to_buffer = options.hindsight_replay_callback(&episodes[i]);
                    if (add_to_buffer) {
                        throw std::runtime_error{"Adding to replay not yet supported."};
                    }
                }
                episodes[i] = rl::agents::dqn::utils::HindsightReplayEpisode{};
                episodes[i].states.push_back(this->states[i]);
            }
        }

        if (options.logger) {
            options.logger->log_frequency("ApexDQN/Inference step rate", options.batch_size);
        }
    }

    void Worker::parse_transition(const rl::utils::reward::NStepCollectorTransition &transition)
    {
        local_buffer->add({
            transition.state->state.to(options.replay_device).unsqueeze(0),
            get_mask(*transition.state->action_constraint).to(options.replay_device).unsqueeze(0),
            transition.action.to(options.replay_device).unsqueeze(0),
            torch::tensor(transition.reward).to(options.replay_device).unsqueeze(0),
            torch::tensor(!transition.terminal).to(options.replay_device).unsqueeze(0),
            transition.next_state->state.to(options.replay_device).unsqueeze(0),
            get_mask(*transition.next_state->action_constraint).to(options.replay_device).unsqueeze(0)
        });

        if (local_buffer->size() >= options.inference_replay_size) {
            replay_buffer->add(*local_buffer->get(torch::arange(options.inference_replay_size)));
            local_buffer->clear();
        }
    }
}
