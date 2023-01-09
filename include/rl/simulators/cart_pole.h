#ifndef RL_SIMULATORS_CART_POLE_H_
#define RL_SIMULATORS_CART_POLE_H_


#include <torch/torch.h>

#include <rl/option.h>

#include "base.h"

namespace rl::simulators
{
    struct CartPoleOptions
    {
        RL_OPTION(bool, sparse_reward) = false;
        RL_OPTION(float, reward_scaling_factor) = 1.0f;
    };

    class ContinuousCartPole : public Base
    {
        public:
            ContinuousCartPole(int steps, const CartPoleOptions &options={});

            States reset(int64_t n) const override;
            Observations step(const torch::Tensor &states, const torch::Tensor &actions) const override;
        
        private:
            const int steps;
            const CartPoleOptions options;
    };

    class DiscreteCartPole : public Base
    {
        public:
            DiscreteCartPole(int steps, int n_actions, const CartPoleOptions &options={});

            States reset(int64_t n) const override;
            Observations step(const torch::Tensor &states, const torch::Tensor &actions) const override;
        
        private:
            const int n_actions;
            torch::Tensor forces;
            ContinuousCartPole sim;
    };
}

#endif /* RL_SIMULATORS_CART_POLE_H_ */
