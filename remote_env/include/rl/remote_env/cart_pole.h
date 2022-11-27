#ifndef RL_ENV_REMOTE_CART_POLE_H_
#define RL_ENV_REMOTE_CART_POLE_H_


#include <string>
#include <memory>

#include <grpc/grpc.h>
#include <grpcpp/grpcpp.h>

#include <rlbuf/remote_env/cart_pole.grpc.pb.h>
#include <rlbuf/remote_env/cart_pole.pb.h>

#include <rl/env/base.h>
#include <rl/policies/constraints/categorical_mask.h>
#include <rl/option.h>

namespace rl::remote_env
{
    struct CartPoleOptions {};


    class CartPole : public rl::env::Base
    {
        public:
            CartPole(
                const std::string &host,
                const CartPoleOptions &options={}
            );

            ~CartPole();

            inline
            bool is_terminal() const override { return terminal; }

            std::unique_ptr<rl::env::State> state() const override;

            std::unique_ptr<rl::env::State> reset() override;

            std::unique_ptr<rl::env::Observation> step(const torch::Tensor &action) override;

        private:
            const CartPoleOptions options;

            grpc::ClientContext client_context{};
            std::unique_ptr<rlbuf::remote_env::cart_pole::CartPoleService::Stub> stub;
            std::unique_ptr<grpc::ClientReaderWriter<rlbuf::remote_env::cart_pole::Action, rlbuf::remote_env::cart_pole::Observation>> pipe;

            bool terminal{true};
            torch::Tensor state_;
            float last_reward{0.0f};
            float total_reward{0.0f};
            std::shared_ptr<rl::policies::constraints::CategoricalMask> constraint;
        
        private:
            void read();
    };


    class CartPoleFactory : public rl::env::Factory
    {
        public:
            CartPoleFactory(
                const std::string &host,
                const CartPoleOptions &options={}
            ) : options{options}, host{host} {}
        
        private:
            const CartPoleOptions options;
            const std::string host;
        
        private:
            std::unique_ptr<rl::env::Base> get_impl() const override {
                return std::make_unique<CartPole>(host, options);
            }
    };
}

#endif /* RL_ENV_REMOTE_CART_POLE_H_ */
