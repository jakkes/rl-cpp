#include "rl/agents/ppo/trainers/basic.h"

#include <thread>

#include <thread_pool.hpp>

#include "rl/buffers/tensor_and_pointer.h"
#include "rl/policies/constraints/base.h"

#include "rl/cpputils.h"


using namespace torch::indexing;
using namespace rl;

using Logger = std::shared_ptr<rl::logging::client::Base>;

namespace rl::agents::ppo::trainers
{
    static
    torch::Tensor compute_policy_loss(torch::Tensor A, torch::Tensor old_probs, torch::Tensor new_probs, float eps)
    {
        auto pr = new_probs / old_probs;
        auto clipped = pr.clamp(1-eps, 1+eps);
        pr = pr * A;
        clipped = clipped * A;
        return - torch::min(clipped, pr).mean();
    }

    static
    torch::Tensor compute_deltas(torch::Tensor rewards, torch::Tensor V, torch::Tensor not_terminals, float discount)
    {
        return rewards + discount * not_terminals * V.index({"...", Slice(1, None)}).detach() - V.index({"...", Slice(None, -1)});
    }

    static
    torch::Tensor compute_advantages(torch::Tensor deltas, torch::Tensor not_terminals, float discount, float gae_discount)
    {
        float d = discount * gae_discount;
        auto A = torch::empty_like(deltas);

        A.index_put_({"...", -1}, deltas.index({"...", -1}));
        for (int k = A.size(1) - 2; k > -1; k--) {
            A.index_put_({"...", k}, deltas.index({"...", k}) + d * not_terminals.index({"...", k}) * A.index({"...", k + 1}));
        }

        return A;
    }

    static
    torch::Tensor compute_value_loss(torch::Tensor deltas)
    {
        return deltas.square().mean();
    }

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
    void run_sequence(std::shared_ptr<env::Base> env, std::shared_ptr<agents::ppo::Module> model, int length, Sequences *out, int out_i, const Logger &logger)
    {
        torch::NoGradGuard no_grad{};
        Sequence sequence{length};

        for (int i = 0; i < length; i++)
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

            if (was_terminal && logger) {
                logger->log_scalar("PPO/StartValue", model_output->value.index({0}).item().toFloat());
                logger->log_scalar("PPO/StartEntropy", model_output->policy->entropy().index({0}).item().toFloat());
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
        int length,
        thread_pool &pool,
        const Logger &logger
    )
    {
        auto re = std::make_unique<Sequences>(envs.size());

        for (int i = 0; i < envs.size(); i++) {
            pool.push_task(run_sequence, envs[i], model, length, re.get(), i, logger);
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
            auto sequences = run_sequences(envs, model, options.sequence_length, pool, options.logger);
            CompiledSequences compiled{*sequences};

            for (int i = 0; i < options.update_steps; i++) {
                auto loss = loss_fn(compiled, model, options);
                optimizer->zero_grad();
                loss.backward();
                optimizer->step();

                if (options.logger) options.logger->log_scalar("PPO/Loss", loss.item().toFloat());
            }
        }
    }

    template void Basic::run<int64_t, std::ratio<1L>>(std::chrono::duration<int64_t, std::ratio<1L>> duration);
}
