#ifndef RL_SIMULATORS_CART_POLE_H_
#define RL_SIMULATORS_CART_POLE_H_


#include <torch/torch.h>

#include "base.h"

namespace rl::simulators
{
    class ContinuousCartPole : public Base
    {
        public:
            ContinuousCartPole();

            States reset(int64_t n) const override;
            Observations step(const torch::Tensor &states, const torch::Tensor &actions) const override;
    };

    class ContinuousCartPoleFactory : public Factory
    {
        public:
            std::unique_ptr<Base> get() const override {
                return std::make_unique<ContinuousCartPole>();
            }
    };

    class DiscreteCartPole : public Base
    {
        public:
            DiscreteCartPole(int n_actions);

            States reset(int64_t n) const override;
            Observations step(const torch::Tensor &states, const torch::Tensor &actions) const override;
        
        private:
            const int n_actions;
            torch::Tensor forces;
            ContinuousCartPole sim{};
    };

    class DiscreteCartPoleFactory : public Factory
    {
        public:
            DiscreteCartPoleFactory(int n_actions) : n_actions{n_actions} {}

            std::unique_ptr<Base> get() const override {
                return std::make_unique<DiscreteCartPole>(n_actions);
            }
        
        private:
            const int n_actions;
    };
}

#endif /* RL_SIMULATORS_CART_POLE_H_ */
