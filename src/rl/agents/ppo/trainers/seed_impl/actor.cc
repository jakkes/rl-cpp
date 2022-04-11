#include "actor.h"

#include <algorithm>


namespace rl::agents::ppo::trainers::seed_impl
{

    Actor::Actor(
        std::shared_ptr<Inference> inference,
        std::shared_ptr<rl::env::Factory> env_factory,
        std::shared_ptr<thread_safe::Queue<std::shared_ptr<Sequence>>> out_stream,
        const ActorOptions &options
    ) :
    inference{inference}, env_factory{env_factory}, out_stream{out_stream}, options{options}
    {}

    void Actor::start()
    {
        if (is_running) throw std::runtime_error{"Actor already started."};
        is_running = true;
        working_thread = std::thread(&Actor::worker, this);
    }

    void Actor::stop()
    {
        is_running = false;
    }

    void Actor::join()
    {
        stop();
        if (working_thread.joinable()) working_thread.join();
    }

    struct Env
    {
        std::shared_ptr<rl::env::Base> env;
        std::unique_ptr<InferenceResultFuture> inference_result;
        std::shared_ptr<Sequence> sequence;
        int sequence_length;
    };

    void Actor::worker()
    {
        std::vector<Env> envs{};
        envs.reserve(options.environments);

        // Create envs
        for (int i = 0; i < options.environments; i++) {
            envs.push_back(
                {
                    env_factory->get(),
                    nullptr,
                    std::make_shared<Sequence>(options.sequence_length),
                    0
                }
            );
        }

        // Initialize while loop
        for (auto &env : envs)
        {
            auto state = env.env->reset();
            env.inference_result = inference->infer(*state);
        }

        int last_env_step_i = options.environments - 1;
        while (is_running)
        {
            int next_env_step_i = (last_env_step_i + 1) % options.environments;

            for (int i = 0; i < options.environments; i++) {
                int j = (i + next_env_step_i) % options.environments;
                if (envs[j].inference_result->is_ready()) {
                    next_env_step_i = j;
                    break;
                }
            }

            auto &env = envs[next_env_step_i];
            last_env_step_i = next_env_step_i;
            
            auto state = env.env->state();
            auto inference_result = env.inference_result->get();
            auto transition = env.env->step(inference_result->action);

            int step = env.sequence_length++;

            env.sequence->states.push_back(state->state);
            env.sequence->constraints.push_back(state->action_constraint);
            env.sequence->state_values.push_back(inference_result->value);
            env.sequence->rewards.push_back(transition->reward);
            env.sequence->actions.push_back(inference_result->action);
            env.sequence->action_probabilities.push_back(inference_result->action_probability);
            env.sequence->not_terminals.push_back(!transition->terminal);

            if (env.sequence_length >= options.sequence_length) {
                env.sequence->states.push_back(transition->state->state);
                env.sequence->constraints.push_back(transition->state->action_constraint);

                out_stream->enqueue(env.sequence);
                env.sequence_length = 0;
                env.sequence = std::make_shared<Sequence>(options.sequence_length);
            }

            auto next_state = transition->state;
            if (transition->terminal) next_state = env.env->reset();

            env.inference_result = inference->infer(*next_state);
        }
    }
}
