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
}

#endif /* RL_SIMULATORS_CART_POLE_H_ */
