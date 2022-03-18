#include "rl/env/cart_pole.h"

#include <random>

#include "rl/policies/constraints/categorical_mask.h"


namespace rl::env
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

    static std::random_device seeder{};
    static auto die = std::bind(
        std::uniform_real_distribution<float>(-0.05, 0.05),
        std::default_random_engine{seeder()}
    );
    

    CartPole::CartPole(int max_steps) : max_steps{max_steps} {}

    std::unique_ptr<State> CartPole::state() {
        auto re = std::make_unique<State>();
        re->state = torch::tensor({x, v, theta, omega});
        re->action_constraint = std::make_shared<rl::policies::constraints::CategoricalMask>(torch::ones({2}, torch::TensorOptions{}.dtype(torch::kBool)));
        return re;
    }

    std::unique_ptr<State> CartPole::reset()
    {
        terminal = false;
        steps = 0;
        x = die();
        v = die();
        theta = die();
        omega = die();
        
        return state();
    }

    std::unique_ptr<Observation> CartPole::step(const torch::Tensor &action) {
        return step(action.item().toLong());
    }

    std::unique_ptr<Observation> CartPole::step(int action)
    {
        if (terminal) throw std::runtime_error{"Cannot step when in a terminal state."};

        float force;
        if (action == 0) force = -FORCE_MAGNITUDE;
        else if (action == 1) force = FORCE_MAGNITUDE;
        else throw std::invalid_argument{"Unknown action " + std::to_string(action)};

        auto costheta = std::cos(theta);
        auto sintheta = std::sin(theta);

        auto temp = (force + POLE_MASS_LENGTH * omega * omega * sintheta) / MASS_TOTAL;
        auto alpha = (G * sintheta - costheta * temp) / (HALF_POLE_LENGTH * (4.0 / 3.0 - MASS_POLE * costheta * costheta / MASS_TOTAL));
        auto a = temp - POLE_MASS_LENGTH * alpha * costheta / MASS_TOTAL;

        x += DT * v;
        v += DT * a;
        theta += DT * omega;
        omega += DT * alpha;

        steps++;
        if (std::abs(x) > POSITION_LIMIT) terminal = true;
        else if (std::abs(theta) > ANGLE_LIMIT) terminal = true;
        else if (steps >= max_steps) terminal = true;

        auto observation = std::make_unique<Observation>();
        observation->reward = 1.0;
        observation->state = state();
        observation->terminal = terminal;

        return observation;
    }

    bool CartPole::is_terminal() { return terminal; }

    CartPoleFactory::CartPoleFactory(int max_steps) : max_steps{max_steps} {}

    std::unique_ptr<Base> CartPoleFactory::get() const 
    {
        return std::make_unique<CartPole>(max_steps);
    }
}