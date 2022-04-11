#include "rl/agents/ppo/trainers/basic.h"

#include <thread>

#include <thread_pool.hpp>

#include "rl/policies/constraints/base.h"
#include "loss_fns.h"


using namespace torch::indexing;
using namespace rl;

using Logger = std::shared_ptr<rl::logging::client::Base>;

namespace rl::agents::ppo::trainers
{

    struct Sequence {
        std::vector<torch::Tensor> states{};
        std::vector<torch::Tensor> actions{};
        std::vector<float> rewards{};
        std::vector<uint8_t> not_terminals{};
        std::vector<torch::Tensor> action_probabilities{};
        std::vector<torch::Tensor> state_values{};
        std::vector<std::shared_ptr<policies::constraints::Base>> constraints{};

        Sequence(int length)
        {
            states.reserve(length + 1);
            actions.reserve(length);
            rewards.reserve(length);
            not_terminals.reserve(length);
            action_probabilities.reserve(length);
            state_values.reserve(length);   // Final state values not necessary
            constraints.reserve(length + 1);
        }
    };

    struct Sequences {
        std::vector<torch::Tensor> states{};
        std::vector<torch::Tensor> actions{};
        std::vector<torch::Tensor> rewards{};
        std::vector<torch::Tensor> not_terminals{};
        std::vector<torch::Tensor> action_probabilities{};
        std::vector<torch::Tensor> state_values{};
        std::vector<std::shared_ptr<policies::constraints::Base>> constraints{};

        Sequences(int batchsize)
        {
            states.resize(batchsize);
            actions.resize(batchsize);
            rewards.resize(batchsize);
            not_terminals.resize(batchsize);
            action_probabilities.resize(batchsize);
            state_values.resize(batchsize);
            constraints.resize(batchsize);
        }

        void set(const Sequence &sequence, int i) {
            assert(i < states.size());
            states[i] = torch::stack(sequence.states);
            actions[i] = torch::stack(sequence.actions);
            rewards[i] = torch::tensor(sequence.rewards, torch::TensorOptions{}.dtype(states[i].dtype()).device(states[i].device()));
            not_terminals[i] = torch::tensor(sequence.not_terminals, torch::TensorOptions{}.dtype(torch::kBool).device(rewards[i].device()));
            action_probabilities[i] = torch::stack(sequence.action_probabilities);
            state_values[i] = torch::stack(sequence.state_values);
            constraints[i] = rl::policies::constraints::stack(sequence.constraints);
        }
    };

    struct CompiledSequences {
        torch::Tensor states{};
        torch::Tensor actions{};
        torch::Tensor rewards{};
        torch::Tensor not_terminals{};
        torch::Tensor action_probabilities{};
        torch::Tensor state_values{};
        std::shared_ptr<policies::constraints::Base> constraints{};

        CompiledSequences(const Sequences &sequences)
        {
            states = torch::stack(sequences.states);
            actions = torch::stack(sequences.actions);
            rewards = torch::stack(sequences.rewards);
            not_terminals = torch::stack(sequences.not_terminals);
            action_probabilities = torch::stack(sequences.action_probabilities);
            state_values = torch::stack(sequences.state_values);
            constraints = rl::policies::constraints::stack(sequences.constraints);
        }
    };

    static
    void run_sequence(std::shared_ptr<env::Base> env, std::shared_ptr<agents::ppo::Module> model, const BasicOptions &options, Sequences *out, int out_i)
    {
        torch::NoGradGuard no_grad{};
        Sequence sequence{options.sequence_length};

        for (int i = 0; i < options.sequence_length; i++)
        {
            auto state = env->state();
            bool was_terminal = env->is_terminal();
            if (was_terminal) state = env->reset();
            sequence.states.push_back(state->state);
            sequence.constraints.push_back(state->action_constraint);

            auto model_output = model->forward(state->state.unsqueeze(0));
            model_output->policy->include(state->action_constraint);
            
            sequence.state_values.push_back(model_output->value.index({0}));

            auto action = model_output->policy->sample().index({0});
            sequence.actions.push_back(action);
            sequence.action_probabilities.push_back(model_output->policy->log_prob(action.unsqueeze(0)).exp().index({0}));

            auto observation = env->step(action);
            sequence.not_terminals.push_back(!observation->terminal);
            sequence.rewards.push_back(observation->reward);

            if (was_terminal && options.logger) {
                if (options.log_start_value) options.logger->log_scalar("PPO/StartValue", model_output->value.index({0}).item().toFloat());
                if (options.log_start_entropy) options.logger->log_scalar("PPO/StartEntropy", model_output->policy->entropy().index({0}).item().toFloat());
            }
        }

        auto final_state = env->state();
        sequence.states.push_back(final_state->state);

        out->set(sequence, out_i);
    }

    static
    std::unique_ptr<Sequences> run_sequences(
        const std::vector<std::shared_ptr<env::Base>> &envs,
        std::shared_ptr<agents::ppo::Module> model,
        const BasicOptions &options,
        thread_pool &pool
    )
    {
        auto re = std::make_unique<Sequences>(envs.size());

        for (int i = 0; i < envs.size(); i++) {
            pool.push_task(run_sequence, envs[i], model, options, re.get(), i);
        }
        pool.wait_for_tasks();
        return re;
    }

    static
    torch::Tensor loss_fn(const CompiledSequences &sequences, const std::shared_ptr<agents::ppo::Module> &model, const BasicOptions &options)
    {
        auto model_output = model->forward(sequences.states.index({Slice(), Slice(None, -1)}));
        model_output->policy->include(sequences.constraints);
        auto action_probabilities = model_output->policy->prob(sequences.actions);

        auto last_state_output = model->forward(sequences.states.index({Slice(), Slice(-1, None)}));
        auto values = torch::cat({model_output->value, last_state_output->value}, 1);

        auto deltas = compute_deltas(sequences.rewards, values, sequences.not_terminals, options.discount);
        auto advantages = compute_advantages(deltas.detach(), sequences.not_terminals, options.discount, options.gae_discount);
        auto value_loss = compute_value_loss(deltas);
        auto policy_loss = compute_policy_loss(advantages, sequences.action_probabilities, action_probabilities, options.eps);

        return value_loss + policy_loss;
    }

    Basic::Basic(
        std::shared_ptr<agents::ppo::Module> model,
        std::shared_ptr<torch::optim::Optimizer> optimizer,
        std::shared_ptr<env::Factory> env_factory,
        const BasicOptions &options
    ) : model{model}, optimizer{optimizer},
        env_factory{env_factory}, options{options}
    {}

    template<class Rep, class Period>
    void Basic::run(std::chrono::duration<Rep, Period> duration)
    {
        auto start = std::chrono::steady_clock::now();
        auto end = start + duration;

        thread_pool pool(options.env_workers);

        std::vector<std::shared_ptr<env::Base>> envs{};
        envs.reserve(options.envs);
        for (int i = 0; i < options.envs; i++) {
            envs.push_back(env_factory->get());
        }

        while (std::chrono::steady_clock::now() < end) {
            auto sequences = run_sequences(envs, model, options, pool);
            CompiledSequences compiled{*sequences};

            for (int i = 0; i < options.update_steps; i++) {
                auto loss = loss_fn(compiled, model, options);
                optimizer->zero_grad();
                loss.backward();
                optimizer->step();

                if (options.logger && options.log_loss) {
                    options.logger->log_scalar("PPO/Loss", loss.item().toFloat());
                }
            }
            if (options.logger) options.logger->log_frequency("PPO/UpdateFrequency", options.update_steps);
        }
    }

    template void Basic::run<int64_t, std::ratio<1L>>(std::chrono::duration<int64_t, std::ratio<1L>> duration);
}
