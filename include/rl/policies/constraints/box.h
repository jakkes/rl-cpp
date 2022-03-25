#ifndef RL_POLICIES_CONSTRAINTS_BOX_H_
#define RL_POLICIES_CONSTRAINTS_BOX_H_

#include <torch/torch.h>

#include "base.h"
#include "rl/cpputils.h"


namespace rl::policies::constraints
{

    struct BoxOptions
    {
        RL_OPTION(bool, inclusive_lower) = true;
        RL_OPTION(bool, inclusive_upper) = true;
    };

    class Box : public Base
    {
        public:
            Box(torch::Tensor lower_bound, torch::Tensor upper_bound, const BoxOptions &options={});

            torch::Tensor contains(const torch::Tensor &x) const override;
            std::function<std::unique_ptr<Base>(const std::vector<std::shared_ptr<Base>>&)> stack_fn() const override;
        
        private:
            const BoxOptions options;
            const torch::Tensor lower, upper;
    };
}

#endif /* RL_POLICIES_CONSTRAINTS_BOX_H_ */
