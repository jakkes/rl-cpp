#ifndef RL_POLICIES_CONSTRAINTS_CAT_H_
#define RL_POLICIES_CONSTRAINTS_CAT_H_


#include <vector>

#include "base.h"


namespace rl::policies::constraints
{
    class Concat : public Base
    {
        public:
            Concat(const std::vector<std::shared_ptr<Base>> &constraints);
            Concat(std::initializer_list<std::shared_ptr<Base>> constraints);
            void push_back(std::shared_ptr<Base> constraint);
            torch::Tensor contains(const torch::Tensor &x) const;

        private:
            std::vector<std::shared_ptr<Base>> constraints;
    };
}

#endif /* RL_POLICIES_CONSTRAINTS_CAT_H_ */
