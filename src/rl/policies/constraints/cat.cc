#include "rl/policies/constraints/cat.h"

#include "rl/cpputils.h"


namespace rl::policies::constraints
{
    Concat::Concat(const std::vector<std::shared_ptr<Base>> &constraints)
    : constraints{constraints}
    {
        if (constraints.empty()) throw std::invalid_argument{"Empty constraint list."};
    }

    Concat::Concat(std::initializer_list<std::shared_ptr<Base>> constraints)
    :  Concat{std::vector<std::shared_ptr<Base>>(constraints.begin(), constraints.end())}
    {}

    void Concat::push_back(std::shared_ptr<Base> constraint)
    {
        constraints.push_back(constraint);
    }

    torch::Tensor Concat::contains(const torch::Tensor &value) const
    {
        auto re = constraints[0]->contains(value);
        
        for (int i = 1; i < constraints.size(); i++) {
            re.logical_and_(constraints[i]->contains(value));
        }

        return re;
    }

    size_t Concat::size() const { return this->constraints.size(); }

    std::function<std::unique_ptr<Base>(const std::vector<std::shared_ptr<Base>>&)> Concat::stack_fn() const
    {
        return __stack_recast<Concat>;
    }

    template<>
    std::unique_ptr<Concat> stack<Concat>(const std::vector<std::shared_ptr<Concat>> &constraints)
    {
        if (constraints.size() == 0) {
            throw std::invalid_argument{"Cannot stack zero elements."};
        }

        auto concat_size = constraints.size();
        std::vector<std::shared_ptr<Base>> stacked_constraints{};
        stacked_constraints.reserve(concat_size);

        for (size_t i = 0; i < concat_size; i++)
        {
            std::vector<std::shared_ptr<Base>> to_be_stacked{};
            to_be_stacked.reserve(constraints.size());
            
            for (auto constraint : constraints) {
                to_be_stacked.push_back(constraint);
            }
            assert(to_be_stacked.size() > 0);
            stacked_constraints.push_back(to_be_stacked[0]->stack_fn()(to_be_stacked));
        }

        return std::make_unique<Concat>(stacked_constraints);
    }
}
