#ifndef RL_ENV_CART_POLE_H_
#define RL_ENV_CART_POLE_H_

#include <cmath>

#include <torch/torch.h>

#include "base.h"

namespace rl::env
{

    class CartPoleContinuous : public Base
    {
        public:
            CartPoleContinuous(int max_steps);

            std::unique_ptr<Observation> step(const torch::Tensor &action) override;
            std::unique_ptr<Observation> step(float action);
            std::unique_ptr<State> state() override;
            std::unique_ptr<State> reset() override;
            bool is_terminal() override;

        protected:
            torch::Tensor state_vector();

        private:
            float x, v, theta, omega;
            bool terminal{true};
            int steps{0};
            float total_reward{0.0};
            const int max_steps;

            void log_terminal();
    };

    class CartPoleContinuousFactory : public Factory
    {
        public:
            CartPoleContinuousFactory(int max_steps, std::shared_ptr<rl::logging::client::Base> logger={});

        private:
            std::unique_ptr<Base> get_impl() const override;
            const int max_steps;
            std::shared_ptr<rl::logging::client::Base> logger;
    };

    class CartPoleDiscrete : public CartPoleContinuous
    {
        public:
            using CartPoleContinuous::CartPoleContinuous;

            std::unique_ptr<State> state() override;
            std::unique_ptr<Observation> step(const torch::Tensor &action) override;
    };

    class CartPoleDiscreteFactory : public Factory
    {
        public:
            CartPoleDiscreteFactory(int max_steps, std::shared_ptr<rl::logging::client::Base> logger={});

        private:
            std::unique_ptr<Base> get_impl() const override;
            const int max_steps;
            std::shared_ptr<rl::logging::client::Base> logger;
    };
}

#endif /* RL_ENV_CART_POLE_H_ */
