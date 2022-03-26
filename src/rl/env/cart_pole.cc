#include "rl/env/cart_pole.h"

#include <random>

#include "rl/policies/constraints/categorical_mask.h"
#include "rl/policies/constraints/box.h"


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
    

    CartPoleContinuous::CartPoleContinuous(int max_steps) : max_steps{max_steps} {}

    torch::Tensor CartPoleContinuous::state_vector() {
        float progress = (steps * 1.0 / max_steps - 0.5) * 2;
        return torch::tensor({x, v, theta, omega, progress}, torch::TensorOptions{}.device(is_cuda() ? torch::kCUDA : torch::kCPU));
    }

    std::unique_ptr<State> CartPoleContinuous::state() {
        auto re = std::make_unique<State>();
        re->state = state_vector();
        auto constraint_options = torch::TensorOptions{}
                                    .device(is_cuda() ? torch::kCUDA : torch::kCPU);
        re->action_constraint = std::make_shared<rl::policies::constraints::Box>(
            torch::tensor(-1.0, constraint_options),
            torch::tensor(1.0, constraint_options)
        );
        return re;
    }

    std::unique_ptr<State> CartPoleContinuous::reset()
    {
        terminal = false;
        steps = 0;
        total_reward = 0.0;
        x = die();
        v = die();
        theta = die();
        omega = die();
        
        return state();
    }

    std::unique_ptr<Observation> CartPoleContinuous::step(const torch::Tensor &action) {
        return step(action.item().toFloat());
    }

    std::unique_ptr<Observation> CartPoleContinuous::step(float action)
    {
        if (terminal) throw std::runtime_error{"Cannot step when in a terminal state."};
        if (action < -1 || action > 1) throw std::invalid_argument{"Action must be in [-1, 1]"};

        float force = action * FORCE_MAGNITUDE;

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

        total_reward += observation->reward;

        if (terminal) log_terminal();

        return observation;
    }

    bool CartPoleContinuous::is_terminal() { return terminal; }

    void CartPoleContinuous::log_terminal() {
        if (!logger) return;
        logger->log_scalar("CartPole/Reward", total_reward);
    }

    std::unique_ptr<State> CartPoleDiscrete::state() {
        auto re = std::make_unique<State>();
        re->state = state_vector();
        re->action_constraint = std::make_shared<rl::policies::constraints::CategoricalMask>(
            torch::ones(
                {2},
                torch::TensorOptions{}
                    .dtype(torch::kBool).device(is_cuda() ? torch::kCUDA : torch::kCPU)
            )
        );
        return re;
    }

    std::unique_ptr<Observation> CartPoleDiscrete::step(const torch::Tensor &action)
    {
        int a = action.item().toLong();
        if (a != 0 && a != 1) throw std::invalid_argument{"Unknown action, must be 0 or 1."};

        return CartPoleContinuous::step(static_cast<float>(a * 2 - 1));
    }

    CartPoleFactory::CartPoleFactory(int max_steps,
                                std::shared_ptr<rl::logging::client::Base> logger)
    : max_steps{max_steps}, logger{logger} {}

    std::unique_ptr<Base> CartPoleFactory::get_impl() const 
    {
        auto env = std::make_unique<CartPoleContinuous>(max_steps);
        env->set_logger(logger);
        return env;
    }
}