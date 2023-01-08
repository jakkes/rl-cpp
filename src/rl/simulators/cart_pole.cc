#include "rl/simulators/cart_pole.h"

#include <rl/policies/constraints/box.h>
#include <rl/policies/constraints/categorical_mask.h>


using namespace torch::indexing;

namespace rl::simulators
{

    static constexpr float POSITION_LIMIT{2.4};
    static constexpr float ANGLE_LIMIT{12 * M_PI / 180};

    static constexpr float G{9.8};
    static constexpr float MASS_CART{1.0};
    static constexpr float MASS_POLE{0.1};
    static constexpr float MASS_TOTAL{MASS_CART + MASS_POLE};
    static constexpr float HALF_POLE_LENGTH{0.5};
    static constexpr float POLE_MASS_LENGTH{MASS_POLE * HALF_POLE_LENGTH};
    static constexpr float FORCE_MAGNITUDE{10.0};
    static constexpr float DT{0.02};

    static inline
    std::shared_ptr<rl::policies::constraints::Box> get_continuous_constraint(int n)
    {
        return std::make_shared<rl::policies::constraints::Box>(
            -torch::ones({n}),
            torch::ones({n}),
            rl::policies::constraints::BoxOptions{}
                .inclusive_lower_(true)
                .inclusive_upper_(true)
        );
    }

    static inline
    std::shared_ptr<rl::policies::constraints::CategoricalMask> get_discrete_constraint(int n, int n_actions)
    {
        static auto tensor_options = torch::TensorOptions{}.dtype(torch::kBool);

        return std::make_shared<rl::policies::constraints::CategoricalMask>(
            torch::ones({n, n_actions}, tensor_options)
        );
    }

    static inline
    torch::Tensor fresh_states(int n) {
        return torch::concat(
            {0.1f * torch::rand({n, 4}) - 0.05f, torch::zeros({n, 1})},
            1
        );
    }

    ContinuousCartPole::ContinuousCartPole(int steps, const CartPoleOptions &options)
    : steps{steps}, options{options}
    {}

    States ContinuousCartPole::reset(int64_t n) const
    {
        States out{};
        out.states = fresh_states(n);
        out.action_constraints = get_continuous_constraint(n);
        return out;
    }

    Observations ContinuousCartPole::step(const torch::Tensor &states,
                                                const torch::Tensor &actions_) const
    {
        if ((actions_ < -1.0f).logical_or(actions_ > 1.0f).any().item().toBool()) {
            throw std::invalid_argument{"Actions must be in [-1.0, 1.0]."};
        }

        auto n = states.size(0);

        auto actions = torch::where(
            (actions_ > -0.2).logical_and(actions_ < 0.2),
            0.2f * actions_.sign(),
            actions_
        );
        actions = torch::where(
            actions == 0.0f,
            0.2f + torch::zeros_like(actions),
            actions
        );

        auto force = FORCE_MAGNITUDE * actions;

        auto x = states.index({Slice(), 0});
        auto v = states.index({Slice(), 1});
        auto theta = states.index({Slice(), 2});
        auto omega = states.index({Slice(), 3});
        auto steps = states.index({Slice(), 4});

        if ((steps >= this->steps).any().item().toBool()) {
            throw std::invalid_argument{"Cannot step a terminal state."};
        }

        auto temp = (force + POLE_MASS_LENGTH * omega.square() * theta.sin()) / MASS_TOTAL;
        auto alpha = (G * theta.sin() - theta.cos() * temp) / (HALF_POLE_LENGTH * (4.0f / 3.0f - MASS_POLE * theta.cos().square() / MASS_TOTAL));
        auto a = temp - POLE_MASS_LENGTH * alpha * theta.cos() / MASS_TOTAL;

        x = x + DT * v;
        v = v + DT * a;
        theta = theta + DT * omega;
        omega = omega + DT * alpha;

        Observations out{};
        out.next_states.states = torch::stack({x, v, theta, omega, steps + 1}, 1);
        out.next_states.action_constraints = get_continuous_constraint(n);

        out.terminals = (
            (x.abs() > POSITION_LIMIT)
            .logical_or(theta.abs() > ANGLE_LIMIT)
            .logical_or(steps + 1 >= this->steps)
        );

        if (options.sparse_reward) {
            out.rewards = torch::where(out.terminals, steps + 1, torch::zeros_like(steps));
        }
        else {
            out.rewards = torch::ones({n});
        }
        out.rewards *= options.reward_scaling_factor;

        return out;
    }

    DiscreteCartPole::DiscreteCartPole(
        int steps, int n_actions, const CartPoleOptions &options
    ) : sim{steps, options}, n_actions{n_actions}
    {
        if (n_actions < 2) {
            throw std::invalid_argument{"Must use at least two actions."};
        }

        forces = torch::linspace(-1.0f, 1.0f, n_actions);
    }

    States DiscreteCartPole::reset(int64_t n) const
    {
        States out{};
        out.states = fresh_states(n);
        out.action_constraints = get_discrete_constraint(n, n_actions);
        return out;
    }

    Observations DiscreteCartPole::step(const torch::Tensor &states,
                                                const torch::Tensor &actions) const
    {
        auto forces = this->forces.index({actions});
        auto observations = sim.step(states, forces);
        observations.next_states.action_constraints = get_discrete_constraint(
            states.size(0), n_actions
        );
        return observations;
    }
}
