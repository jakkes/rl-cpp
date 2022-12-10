#include "rl/simulators/combinatorial_lock.h"

#include <algorithm>

#include <rl/policies/constraints/categorical_mask.h>


namespace rl::simulators
{
    static auto long_dtype = torch::TensorOptions{}.dtype(torch::kLong);
    static auto bool_dtype = torch::TensorOptions{}.dtype(torch::kBool);

    CombinatorialLock::CombinatorialLock(
        int dim,
        const std::vector<int> &correct_sequence,
        const CombinatorialLockOptions &options
    )
    : dim{dim}, options{options}
    {
        auto non_negative = [] (int x) { return x >= 0; };
        auto less_than_dim = [dim] (int x) { return x < dim; };

        if (!std::all_of(correct_sequence.cbegin(), correct_sequence.cend(), non_negative)) {
            throw std::invalid_argument{"Sequence values must be non negative."};
        }
        if (!std::all_of(correct_sequence.cbegin(), correct_sequence.cend(), less_than_dim)) {
            throw std::invalid_argument{"Sequence values must be less than the given dim."};
        }

        this->correct_sequence = torch::tensor(correct_sequence, long_dtype);
    }

    States CombinatorialLock::reset(int64_t n) const
    {
        States out{};
        out.states = torch::zeros({n, correct_sequence.size(0)}, long_dtype) - 1;
        out.action_constraints = std::make_shared<rl::policies::constraints::CategoricalMask>(
            torch::ones({n, dim}, bool_dtype)
        );
        return out;
    }

    Observations CombinatorialLock::step(const torch::Tensor &states,
                                            const torch::Tensor &actions) const
    {
        if ((actions >= dim).any().item().toBool()) {
            throw std::invalid_argument{"Action must not be larger than dim."};
        }
        if ((actions < 0).any().item().toBool()) {
            throw std::invalid_argument{"Action must not be negative."};
        }

        auto sequence_lengths = (states >= 0).sum(-1);
        if ((sequence_lengths >= correct_sequence.size(0)).any().item().toBool()) {
            throw std::runtime_error{"Cannot step a terminal state."};
        }
        
        auto batchvec = torch::arange(states.size(0));
        
        Observations out{};
        out.next_states.states = states.index_put({batchvec, sequence_lengths}, actions);
        out.next_states.action_constraints = std::make_shared<rl::policies::constraints::CategoricalMask>(
            torch::ones({states.size(0), dim}, bool_dtype)
        );

        if (options.intermediate_rewards) {
            out.rewards = ((out.next_states.states == correct_sequence).sum(-1) > sequence_lengths).to(torch::kFloat32) / correct_sequence.size(0);
        }
        else {
            out.rewards = (out.next_states.states == correct_sequence).all(-1).to(torch::kFloat32);
        }
        out.terminals = sequence_lengths + 1 >= correct_sequence.size(0);
        return out;
    }
}
