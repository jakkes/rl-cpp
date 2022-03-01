#include "rl/buffers/state_transition.h"


namespace rl::buffers
{
    StateTransition::StateTransition(int64_t capacity, const StateTransitionOptions &options)
    : capacity{capacity}, options{options}
    {
        states.resize(capacity);
        rewards = torch::zeros(
            {capacity},
            torch::TensorOptions{}
                .dtype(torch::kFloat32)
                .device(options.device)
        );
        terminals = torch::ones(
            {capacity},
            torch::TensorOptions{}
                .dtype(torch::kBool)
                .device(options.device)
        );
        next_states.resize(capacity);
    }

    int64_t StateTransition::size() const
    {
        if (looped) return capacity;
        return memory_index;
    }

    void StateTransition::add(
        std::shared_ptr<rl::env::State> state,
        float reward,
        bool terminal,
        std::shared_ptr<rl::env::State> next_state
    ) {
        add(
            std::vector<std::shared_ptr<rl::env::State>>{state},
            {reward},
            {terminal},
            {next_state}
        );
    }

    void StateTransition::add(
        const std::vector<std::shared_ptr<rl::env::State>> &states,
        const std::vector<float> &rewards,
        const std::vector<bool> &terminals,
        const std::vector<std::shared_ptr<rl::env::State>> &next_states
    ) {
        std::lock_guard<std::mutex> lock_guard{lock};

        auto bs = states.size();
        assert(rewards.size() == bs);
        assert(terminals.size() == bs);
        assert(next_states.size() == bs);

        int64_t memory_index{this->memory_index};
        for (int i = 0; i < bs; i++, memory_index++) {
            if (memory_index >= capacity) memory_index = 0;
            this->states[memory_index] = states[i];
            this->next_states[memory_index] = next_states[i];
        }

        auto indices = (torch::arange(static_cast<int>(bs)) + static_cast<int64_t>(this->memory_index)) % capacity;
        this->rewards.index_put_(
            {indices},
            torch::tensor(rewards, torch::TensorOptions{}
                .dtype(torch::kFloat32).device(options.device))
        );

        std::vector<int8_t> converted_terminals{};
        converted_terminals.reserve(bs);
        for (const auto &v : terminals) converted_terminals.push_back(static_cast<int8_t>(v));
        this->terminals.index_put_(
            {indices},
            torch::tensor(
                converted_terminals,
                torch::TensorOptions{}.dtype(torch::kBool).device(options.device))
        );

        this->memory_index = this->memory_index + bs;
        if (this->memory_index > capacity) {
            this->memory_index = this->memory_index % capacity;
            looped = true;
        }
    }

    std::unique_ptr<StateTransitionSample> StateTransition::sample(int64_t n)
    {
        std::lock_guard<std::mutex> lock_guard{lock};
        auto indices = torch::randint(size(), {n}, torch::TensorOptions{}.dtype(torch::kLong).device(options.device));

        auto re = std::make_unique<StateTransitionSample>();
        re->rewards = rewards.index({indices});
        re->terminals = terminals.index({indices});

        std::vector<torch::Tensor> states, next_states;
        std::vector<std::shared_ptr<rl::policies::constraints::Base>> constraints, next_constraints;
        states.reserve(n); next_states.reserve(n);
        constraints.reserve(n); next_constraints.reserve(n);

        for (int64_t i = 0; i < n; i++) {
            auto idx = indices.index({i}).item().toLong();
            states.push_back(this->states[idx]->state);
            constraints.push_back(this->states[idx]->action_constraint);
            next_states.push_back(this->next_states[idx]->state);
            next_constraints.push_back(this->next_states[idx]->action_constraint);
        }

        re->states = std::make_shared<rl::env::State>();
        re->states->state = torch::stack(states);
        re->states->action_constraint = constraints[0]->stack(constraints);
        re->next_states = std::make_shared<rl::env::State>();
        re->next_states->state = torch::stack(next_states);
        re->next_states->action_constraint = next_constraints[0]->stack(next_constraints);

        return re;
    }
}
