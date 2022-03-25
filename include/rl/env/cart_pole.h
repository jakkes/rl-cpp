#ifndef RL_ENV_CART_POLE_H_
#define RL_ENV_CART_POLE_H_

#include <cmath>

#include <torch/torch.h>

#include "base.h"

namespace rl::env
{
    class CartPole : public Base
    {
        public:
            CartPole(int max_steps);

            std::unique_ptr<Observation> step(int action);
            std::unique_ptr<Observation> step(const torch::Tensor &action);
            std::unique_ptr<State> state();
            std::unique_ptr<State> reset();
            bool is_terminal();

        private:
            float x, v, theta, omega;
            bool terminal{true};
            int steps{0};
            float total_reward{0.0};
            const int max_steps;

            void log_terminal();
    };

    class CartPoleFactory : public Factory
    {
        public:
            CartPoleFactory(int max_steps, std::shared_ptr<rl::logging::client::Base> logger={});

        private:
            std::unique_ptr<Base> get_impl() const override;
            const int max_steps;
            std::shared_ptr<rl::logging::client::Base> logger;
    };
}

#endif /* RL_ENV_CART_POLE_H_ */
