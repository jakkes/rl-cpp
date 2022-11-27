#include "rl/remote_env/cart_pole.h"


namespace rl::remote_env
{
    CartPole::CartPole(
        const std::string &host,
        const CartPoleOptions &options
    ) : options{options}
    {
        auto channel = grpc::CreateChannel(host, grpc::InsecureChannelCredentials());
        stub = rlbuf::remote_env::cart_pole::CartPoleService::NewStub(channel);
        pipe = stub->EnvStream(&client_context);

        constraint = std::make_shared<rl::policies::constraints::CategoricalMask>(torch::tensor({true, true}));

        read();
    }

    CartPole::~CartPole()
    {
        auto done_result = pipe->WritesDone();
        assert(done_result);

        auto status = pipe->Finish();
        assert(status.ok());
    }

    std::unique_ptr<rl::env::State> CartPole::state() const
    {
        auto out = std::make_unique<rl::env::State>();
        out->state = state_;
        out->action_constraint = constraint;
        return out;
    }

    std::unique_ptr<rl::env::State> CartPole::reset()
    {
        terminal = false;
        total_reward = 0.0f;
        return state();
    }

    void CartPole::read()
    {
        rlbuf::remote_env::cart_pole::Observation observation{};
        auto read_success = pipe->Read(&observation);
        if (!read_success) {
            throw std::runtime_error{"Failed reading from pipe. Is server running?"};
        }

        state_ = torch::tensor({
            observation.next_state().position(),
            observation.next_state().velocity(),
            observation.next_state().angle(),
            observation.next_state().angular_velocity()
        });
        terminal = observation.terminal();
        last_reward = observation.reward();
        total_reward += last_reward;
    }

    std::unique_ptr<rl::env::Observation> CartPole::step(const torch::Tensor &action)
    {
        rlbuf::remote_env::cart_pole::Action content{};
        content.set_action(action.item().toLong());

        auto write_success = pipe->Write(content);
        if (!write_success) {
            throw std::runtime_error{"Failed writing to pipe."};
        }

        read();

        auto out = std::make_unique<rl::env::Observation>();
        out->state = state();
        out->reward = last_reward;
        out->terminal = terminal;

        if (terminal) {
            if (logger) {
                logger->log_scalar("RemoteCartPole/Reward", total_reward);
            }
        }

        return out;
    }
}
