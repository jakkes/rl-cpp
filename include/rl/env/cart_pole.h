#ifndef RL_ENV_CART_POLE_H_
#define RL_ENV_CART_POLE_H_

#include <cmath>

#include <torch/torch.h>

#include "base.h"

namespace rl::env
{

    /**
     * @brief CartPole (inverted pendulum) environment, with a continuous action space.
     * 
     * Actions are floats in [-1, 1].
     * 
     */
    class CartPoleContinuous : public Base
    {
        public:
            /**
             * @brief Construct a new Cart Pole Continuous object
             * 
             * @param max_steps Maximum number of steps in one episode.
             */
            CartPoleContinuous(int max_steps);

            /// <inheritdoc/>
            std::unique_ptr<Observation> step(const torch::Tensor &action) override;
            
            /**
             * @brief Applies one environmental step.
             * 
             * @param action Action, float in [-1, 1]
             * @return std::unique_ptr<Observation> State transition.
             */
            std::unique_ptr<Observation> step(float action);

            /// <inheritdoc/>
            std::unique_ptr<State> state() const override;

            /// <inheritdoc/>
            std::unique_ptr<State> reset() override;

            /// <inheritdoc/>
            bool is_terminal() const override;

        protected:
            torch::Tensor state_vector() const;

        private:
            float x, v, theta, omega;
            bool terminal{true};
            int steps{0};
            float total_reward{0.0};
            const int max_steps;

            void log_terminal();
    };

    /**
     * @brief Environment factory that spawns `CartPoleContinuous` environment
     * instances.
     * 
     */
    class CartPoleContinuousFactory : public Factory
    {
        public:
            /**
             * @brief Construct a new Cart Pole Continuous Factory object
             * 
             * @param max_steps Maximum number of steps per episode in environment
             * instances.
             * @param logger Logger instance that is used by spawned instances.
             */
            CartPoleContinuousFactory(int max_steps);

        private:
            std::unique_ptr<Base> get_impl() const override;
            const int max_steps;
    };

    /**
     * @brief CartPole (inverted pendulum) with a discrete action space.
     * 
     * Action space is discretized into a given number of points.
     * 
     */
    class CartPoleDiscrete : public CartPoleContinuous
    {
        public:
            /**
             * @brief Construct a new Cart Pole Discrete object
             * 
             * @param max_steps Maximum number of steps per episode.
             * @param action_space_dim Number of actions, i.e. size of discretization of
             * [-1, 1].
             */
            CartPoleDiscrete(int max_steps, int action_space_dim);

            /// <inheritdoc/>
            std::unique_ptr<State> state() const override;
            
            /// <inheritdoc/>
            std::unique_ptr<Observation> step(const torch::Tensor &action) override;

        private:
            const int action_space_dim;
            std::vector<float> actions;
    };

    /**
     * @brief Environment factory that spawns CartPoleDiscrete instances.
     * 
     */
    class CartPoleDiscreteFactory : public Factory
    {
        public:
            /**
             * @brief Construct a new Cart Pole Discrete Factory object
             * 
             * @param max_steps Maximum number of steps per episode.
             * @param action_space_dim Size of (discrete) action space.
             * @param logger Logger instance to be used by the spawn environment
             * instances.
             */
            CartPoleDiscreteFactory(int max_steps, int action_space_dim, std::shared_ptr<rl::logging::client::Base> logger={});

        private:
            std::unique_ptr<Base> get_impl() const override;
            const int max_steps, action_space_dim;
            std::shared_ptr<rl::logging::client::Base> logger;
    };
}

#endif /* RL_ENV_CART_POLE_H_ */
