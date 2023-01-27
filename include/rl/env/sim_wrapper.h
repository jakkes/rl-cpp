#ifndef RL_ENV_SIM_WRAPPER_H_
#define RL_ENV_SIM_WRAPPER_H_


#include <memory>

#include <rl/simulators/base.h>
#include "base.h"

namespace rl::env
{
    class SimWrapper : public Base
    {
        public:
            inline
            SimWrapper(std::shared_ptr<rl::simulators::Base> sim) : sim{sim} {}
        
            inline
            std::unique_ptr<Observation> step(const torch::Tensor &action) override
            {
                if (must_reset) {
                    throw std::runtime_error{"Cannot step in a terminal state."};
                }

                auto step = sim->step(state_.unsqueeze(0), action.unsqueeze(0));
                state_ = step.next_states.states.squeeze_(0);
                action_constraint = step.next_states.action_constraints->index({0});

                auto out = std::make_unique<Observation>();
                out->reward = step.rewards.squeeze(0).item().toFloat();
                out->state = state();
                out->terminal = step.terminals.squeeze(0).item().toBool();

                return out;
            }

            inline
            std::unique_ptr<State> reset() override
            {
                must_reset = false;
                auto states = sim->reset(1);
                state_ = states.states.squeeze_(0);
                action_constraint = states.action_constraints->index({0});

                return state();
            }

            inline
            std::unique_ptr<State> state() const override
            {
                auto out = std::make_unique<State>();
                out->state = state_;
                out->action_constraint = action_constraint;

                return out;
            }

            inline
            bool is_terminal() const override { return must_reset; }

        private:
            bool must_reset{true};
            std::shared_ptr<rl::simulators::Base> sim;
            torch::Tensor state_;
            std::shared_ptr<rl::policies::constraints::Base> action_constraint;
    };

    class SimWrapperFactory : public Factory
    {
        public:
            SimWrapperFactory(std::shared_ptr<rl::simulators::Base> sim) : sim{sim} {}

        private:
            std::shared_ptr<rl::simulators::Base> sim;

        private:
            std::unique_ptr<Base> get_impl() const override {
                return std::make_unique<SimWrapper>(sim);
            }
    };
}

#endif /* RL_ENV_SIM_WRAPPER_H_ */
