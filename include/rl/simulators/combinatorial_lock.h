#ifndef RL_SIMULATORS_COMBINATORICAL_LOCK_H_
#define RL_SIMULATORS_COMBINATORICAL_LOCK_H_

#include <vector>

#include <torch/torch.h>

#include <rl/option.h>

#include "base.h"

namespace rl::simulators
{
    struct CombinatorialLockOptions
    {
        RL_OPTION(bool, intermediate_rewards) = false;
    };

    class CombinatorialLock : public Base
    {
        public:
            CombinatorialLock(
                int dim,
                const std::vector<int> &correct_sequence,
                const CombinatorialLockOptions &options={}
            );
            ~CombinatorialLock() = default;

            States reset(int64_t n) const override;
            Observations step(const torch::Tensor &states,
                                const torch::Tensor &actions) const override;

        private:
            const CombinatorialLockOptions options;
            int dim;
            torch::Tensor correct_sequence;
    };
}

#endif /* RL_SIMULATORS_COMBINATORICAL_LOCK_H_ */
