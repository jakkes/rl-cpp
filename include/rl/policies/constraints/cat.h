#ifndef RL_POLICIES_CONSTRAINTS_CAT_H_
#define RL_POLICIES_CONSTRAINTS_CAT_H_


#include <vector>

#include "base.h"
#include "stack.h"


namespace rl::policies::constraints
{
    class Concat : public Base
    {
        public:
            Concat(const std::vector<std::shared_ptr<Base>> &constraints);
            Concat(std::initializer_list<std::shared_ptr<Base>> constraints);
            torch::Tensor contains(const torch::Tensor &x) const;
            std::function<std::unique_ptr<Base>(const std::vector<std::shared_ptr<Base>>&)> stack_fn() const;
            void push_back(std::shared_ptr<Base> constraint);
            size_t size() const;
        private:
            std::vector<std::shared_ptr<Base>> constraints;
    };

    template<>
    std::unique_ptr<Concat> stack<Concat>(const std::vector<std::shared_ptr<Concat>> &constraints);
}

#endif /* RL_POLICIES_CONSTRAINTS_CAT_H_ */
