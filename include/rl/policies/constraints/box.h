#ifndef RL_POLICIES_CONSTRAINTS_BOX_H_
#define RL_POLICIES_CONSTRAINTS_BOX_H_

#include <torch/torch.h>

#include "base.h"
#include "stack.h"
#include "rl/cpputils.h"


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

            torch::Tensor contains(const torch::Tensor &x) const override;

            friend std::unique_ptr<Box> __stack_impl<Box>(const std::vector<std::shared_ptr<Box>> &constraints);
        private:
            const BoxOptions options;
            const torch::Tensor lower, upper;
    };

    template<>
    std::unique_ptr<Box> __stack_impl<Box>(const std::vector<std::shared_ptr<Box>> &constraints);
}

#endif /* RL_POLICIES_CONSTRAINTS_BOX_H_ */
