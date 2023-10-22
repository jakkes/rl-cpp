#ifndef RL_ENV_REMOTE_LUNAR_LANDER_H_
#define RL_ENV_REMOTE_LUNAR_LANDER_H_


#include <string>
#include <memory>

#include <grpc/grpc.h>
#include <grpcpp/grpcpp.h>

#include <rlbuf/remote_env/lunar_lander.grpc.pb.h>
#include <rlbuf/remote_env/lunar_lander.pb.h>

#include <rl/env/base.h>
#include <rl/policies/constraints/categorical_mask.h>
#include <rl/option.h>

namespace rl::remote_env
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
            std::unique_ptr<rlbuf::remote_env::lunar_lander::LunarLanderService::Stub> stub;
            std::unique_ptr<grpc::ClientReaderWriter<rlbuf::remote_env::lunar_lander::Action, rlbuf::remote_env::lunar_lander::Observation>> pipe;

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
                int base_port,
                int number_of_hosts,
                const LunarLanderOptions &options={}
            ) : 
                options{options}, host{host},
                base_port{base_port}, number_of_hosts{number_of_hosts}
            {}
        
        private:
            const LunarLanderOptions options;
            const std::string host;
            const int base_port;
            const int number_of_hosts;
            mutable int i = -1;    
        
        private:
            std::unique_ptr<rl::env::Base> get_impl() const override {
                i = (i + 1) % number_of_hosts;
                auto port = base_port + i;
                auto host_port = host + ":" + std::to_string(port);

                return std::make_unique<LunarLander>(host_port, options);
            }
    };
}

#endif /* RL_ENV_REMOTE_LUNAR_LANDER_H_ */
