#include "rl/env/remote/continuous_lunar_lander.h"


namespace rl::env::remote
{
    ContinuousLunarLander::ContinuousLunarLander(
        const std::string &host,
        const ContinuousLunarLanderOptions &options
    ) : options{options}
    {
        auto channel = grpc::CreateChannel(host, grpc::InsecureChannelCredentials());
        stub = rlbuf::env::remote::continuous_lunar_lander::ContinuousLunarLanderService::NewStub(channel);
        pipe = stub->EnvStream(&client_context);

        constraint = std::make_shared<rl::policies::constraints::Box>(
            - torch::ones({2}),
            torch::ones({2}),
            rl::policies::constraints::BoxOptions{}
                .inclusive_lower_(true)
                .inclusive_upper_(true)
                .n_action_dims_(1)
        );

        read();
    }

    ContinuousLunarLander::~ContinuousLunarLander()
    {
        auto done_result = pipe->WritesDone();
        assert(done_result);

        auto status = pipe->Finish();
        assert(status.ok());
    }

    std::unique_ptr<rl::env::State> ContinuousLunarLander::state() const
    {
        auto out = std::make_unique<rl::env::State>();
        out->state = state_;
        out->action_constraint = constraint;
        return out;
    }

    std::unique_ptr<rl::env::State> ContinuousLunarLander::reset()
    {
        terminal = false;
        total_reward = 0.0f;
        return state();
    }

    void ContinuousLunarLander::read()
    {
        rlbuf::env::remote::continuous_lunar_lander::Observation observation{};
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

    std::unique_ptr<rl::env::Observation> ContinuousLunarLander::step(const torch::Tensor &action)
    {
        rlbuf::env::remote::continuous_lunar_lander::Action content{};
        if (!constraint->contains(action).item().toBool()) {
            throw std::runtime_error{"Action not valid."};
        }
        auto action_accessor = action.accessor<float, 1>();
        content.set_main_engine(action_accessor[0]);
        content.set_lateral_engine(action_accessor[1]);

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
                logger->log_scalar("RemoteContinuousLunarLander/Reward", total_reward);
                logger->log_frequency("RemoteContinuousLunarLander/Episode frequency", 1);
            }
        }

        return out;
    }
}
