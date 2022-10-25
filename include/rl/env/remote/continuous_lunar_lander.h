#ifndef RL_ENV_REMOTE_CONTINUOUS_continuous_lunar_lander_H_
#define RL_ENV_REMOTE_CONTINUOUS_continuous_lunar_lander_H_

#include <grpc/grpc.h>
#include <grpcpp/grpcpp.h>

#include <rl/env/base.h>
#include <rl/policies/constraints/box.h>
#include <rlbuf/env/remote/continuous_lunar_lander.grpc.pb.h>


namespace rl::env::remote
{
    struct ContinuousLunarLanderOptions {};


    class ContinuousLunarLander : public rl::env::Base
    {
        public:
            ContinuousLunarLander(
                const std::string &host,
                const ContinuousLunarLanderOptions &options={}
            );

            ~ContinuousLunarLander();

            inline
            bool is_terminal() const override { return terminal; }

            std::unique_ptr<rl::env::State> state() const override;

            std::unique_ptr<rl::env::State> reset() override;

            std::unique_ptr<rl::env::Observation> step(const torch::Tensor &action) override;

        private:
            const ContinuousLunarLanderOptions options;

            grpc::ClientContext client_context{};
            std::unique_ptr<rlbuf::env::remote::continuous_lunar_lander::ContinuousLunarLanderService::Stub> stub;
            std::unique_ptr<grpc::ClientReaderWriter<rlbuf::env::remote::continuous_lunar_lander::Action, rlbuf::env::remote::continuous_lunar_lander::Observation>> pipe;

            bool terminal{true};
            torch::Tensor state_;
            float last_reward{0.0f};
            float total_reward{0.0f};
            std::shared_ptr<rl::policies::constraints::Box> constraint;
        
        private:
            void read();
    };


    class ContinuousLunarLanderFactory : public rl::env::Factory
    {
        public:
            ContinuousLunarLanderFactory(
                const std::string &host,
                const ContinuousLunarLanderOptions &options={}
            ) : options{options}, host{host} {}
        
        private:
            const ContinuousLunarLanderOptions options;
            const std::string host;
        
        private:
            std::unique_ptr<rl::env::Base> get_impl() const override {
                return std::make_unique<ContinuousLunarLander>(host, options);
            }
    };
}

#endif /* RL_ENV_REMOTE_CONTINUOUS_continuous_lunar_lander_H_ */
