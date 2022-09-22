#ifndef RL_POLICIES_CONSTRAINTS_BOX_H_
#define RL_POLICIES_CONSTRAINTS_BOX_H_


#include <torch/torch.h>

#include "rl/option.h"

#include "base.h"
#include "stack.h"


namespace rl::policies::constraints
{

    struct BoxOptions
    {
        RL_OPTION(bool, inclusive_lower) = true;
        RL_OPTION(bool, inclusive_upper) = true;
        RL_OPTION(int, n_action_dims) = 0;

        bool equals(const BoxOptions &other) const;
    };

    class Box : public Base
    {
        public:
            Box(torch::Tensor lower_bound, torch::Tensor upper_bound,
                                                    const BoxOptions &options={});

            ~Box() = default;

            torch::Tensor contains(const torch::Tensor &x) const override;
            std::unique_ptr<Base> index(const std::vector<torch::indexing::TensorIndex> &indexing) const override;

            std::unique_ptr<Box> stack(const std::vector<std::shared_ptr<Box>> &constraints) const;
            std::unique_ptr<Base> stack(const std::vector<std::shared_ptr<Base>> &constraints) const override;

            const torch::Tensor upper_bound() const;
            const torch::Tensor lower_bound() const;

            int n_action_dims() const;
        private:
            const BoxOptions options;
            const torch::Tensor lower, upper;
    };
}

#endif /* RL_POLICIES_CONSTRAINTS_BOX_H_ */
