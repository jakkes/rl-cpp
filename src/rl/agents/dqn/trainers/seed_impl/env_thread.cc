#include "env_thread.h"

#include "helpers.h"

namespace seed_impl
{
    EnvThread::EnvThread(
        std::shared_ptr<rl::env::Factory> env_factory,
        std::shared_ptr<Inferer> inferer,
        std::shared_ptr<thread_safe::Queue<rl::utils::NStepCollectorTransition>> transition_queue,
        const rl::agents::dqn::trainers::SEEDOptions &options
    ) :
        options{options},
        env_factory{env_factory},
        inferer{inferer},
        transition_queue{transition_queue}
    {}

    void EnvThread::start() {
        running = true;
        worker_thread = std::thread(&EnvThread::worker, this);
    }

    void EnvThread::stop() {
        running = false;
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    void EnvThread::worker()
    {
        workers.reserve(options.envs_per_worker);
        for (int i = 0; i < options.envs_per_worker; i++) {
            workers.emplace_back(env_factory, inferer, transition_queue, options);
        }

        int i = 0;

        while (running)
        {
            i = i % workers.size();

            if (workers[i].ready()) {
                workers[i++].step();
                continue;
            }

            bool stepped{false};
            for (int j = 0; j < workers.size(); j++) {
                int k = (j + i) % workers.size();
                if (workers[k].ready()) {
                    workers[k].step();
                    stepped = true;
                    break;
                }
            }

            if (stepped) {
                continue;
            }

            workers[i++].step();
        }
    }

    EnvWorker::EnvWorker(
        std::shared_ptr<rl::env::Factory> env_factory,
        std::shared_ptr<Inferer> inferer,
        std::shared_ptr<thread_safe::Queue<rl::utils::NStepCollectorTransition>> transition_queue,
        const rl::agents::dqn::trainers::SEEDOptions &options
    ) : options{options}, n_step_collector{options.n_step, options.discount}
    {
        this->inferer = inferer;
        env = env_factory->get();
        this->transition_queue = transition_queue;

        state = env->reset();
        result_future = inferer->infer(state->state, get_mask(*state->action_constraint));
    }

    void EnvWorker::step()
    {
        auto result = result_future->result();
        if (start_state) {
            start_state = false;
            if (options.logger) {
                options.logger->log_scalar("SEEDDQN/Start value", result->value.max().item().toFloat());
            }
        }

        auto observation = env->step(result->action.to(options.environment_device));
        auto transitions = n_step_collector.step(state, result->action, observation->reward, observation->terminal);

        for (const auto &transition : transitions) {
            transition_queue->enqueue(transition);
        }

        state = observation->state;
        if (observation->terminal) {
            state = env->reset();
            start_state = true;
            if (options.logger) {
                options.logger->log_scalar("SEEDDQN/End value", result->value.max().item().toFloat());
            }
        }

        result_future = inferer->infer(state->state, get_mask(*state->action_constraint));
    }
}
