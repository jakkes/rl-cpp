#include "rl/env/remote/lunar_lander.h"


namespace rl::env::remote
{
    LunarLander::LunarLander(
        const std::string &host,
        const LunarLanderOptions &options
    ) : options{options}
    {
        auto channel = grpc::CreateChannel(host, grpc::InsecureChannelCredentials());
        stub = rlbuf::env::remote::lunar_lander::LunarLanderService::NewStub(channel);
        pipe = stub->EnvStream(&client_context);

        constraint = std::make_shared<rl::policies::constraints::CategoricalMask>(torch::tensor({true, true, true, true}));

        read();
    }

    LunarLander::~LunarLander()
    {
        auto done_result = pipe->WritesDone();
        assert(done_result);

        auto status = pipe->Finish();
        assert(status.ok());
    }

    std::unique_ptr<rl::env::State> LunarLander::state() const
    {
        auto out = std::make_unique<rl::env::State>();
        out->state = state_;
        out->action_constraint = constraint;
        return out;
    }

    std::unique_ptr<rl::env::State> LunarLander::reset()
    {
        terminal = false;
        total_reward = 0.0f;
        return state();
    }

    void LunarLander::read()
    {
        rlbuf::env::remote::lunar_lander::Observation observation{};
        auto read_success = pipe->Read(&observation);
        if (!read_success) {
            throw std::runtime_error{"Failed reading from pipe. Is server running?"};
        }

        state_ = torch::tensor(std::vector<float>{
            observation.next_state().data().begin(),
            observation.next_state().data().end()
        });
        terminal = observation.terminal();
        last_reward = observation.reward();
        total_reward += last_reward;
    }

    std::unique_ptr<rl::env::Observation> LunarLander::step(const torch::Tensor &action)
    {
        rlbuf::env::remote::lunar_lander::Action content{};
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
                logger->log_scalar("RemoteLunarLander/Reward", total_reward);
            }
        }

        return out;
    }
}
