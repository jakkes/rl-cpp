#ifndef RL_ENV_REMOTE_LUNAR_LANDER_H_
#define RL_ENV_REMOTE_LUNAR_LANDER_H_


#include <string>
#include <memory>

#include <grpc/grpc.h>
#include <grpcpp/grpcpp.h>

#include <rlbuf/env/remote/lunar_lander.grpc.pb.h>
#include <rlbuf/env/remote/lunar_lander.pb.h>

#include <rl/env/base.h>
#include <rl/policies/constraints/categorical_mask.h>
#include <rl/option.h>

namespace rl::env::remote
{
    struct LunarLanderOptions {};


    class LunarLander : public rl::env::Base
    {
        public:
            LunarLander(
                const std::string &host,
                const LunarLanderOptions &options={}
            );

            ~LunarLander();

            inline
            bool is_terminal() const override { return terminal; }

            std::unique_ptr<rl::env::State> state() const override;

            std::unique_ptr<rl::env::State> reset() override;

            std::unique_ptr<rl::env::Observation> step(const torch::Tensor &action) override;

        private:
            const LunarLanderOptions options;

            grpc::ClientContext client_context{};
            std::unique_ptr<rlbuf::env::remote::lunar_lander::LunarLanderService::Stub> stub;
            std::unique_ptr<grpc::ClientReaderWriter<rlbuf::env::remote::lunar_lander::Action, rlbuf::env::remote::lunar_lander::Observation>> pipe;

            bool terminal{true};
            torch::Tensor state_;
            float last_reward{0.0f};
            float total_reward{0.0f};
            std::shared_ptr<rl::policies::constraints::CategoricalMask> constraint;
        
        private:
            void read();
    };


    class LunarLanderFactory : public rl::env::Factory
    {
        public:
            LunarLanderFactory(
                const std::string &host,
                const LunarLanderOptions &options={}
            ) : options{options}, host{host} {}
        
        private:
            const LunarLanderOptions options;
            const std::string host;
        
        private:
            std::unique_ptr<rl::env::Base> get_impl() const override {
                return std::make_unique<LunarLander>(host, options);
            }
    };
}

#endif /* RL_ENV_REMOTE_LUNAR_LANDER_H_ */
