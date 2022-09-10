#include <rl/rl.h>
#include <torchdebug.h>


using namespace rl;


class Module : public agents::dqn::modules::DQN
{
    public:
        Module();
        std::unique_ptr<agents::dqn::modules::DQNOutput> forward_impl(const torch::Tensor &states) override;
        std::unique_ptr<agents::dqn::modules::Base> clone() const override;
    
    private:
        torch::nn::Sequential net;
};


int main()
{
    auto logger = std::make_shared<logging::client::EMA>(std::initializer_list<double>{0.6, 0.9, 0.99, 0.999}, 1);
    auto env_factory = std::make_shared<rl::env::CartPoleDiscreteFactory>(200, 2, logger);
    env_factory->set_logger(logger);
    auto model = std::make_shared<Module>();
    auto optimizer = std::make_shared<torch::optim::Adam>(model->parameters());
    auto policy = std::make_shared<agents::dqn::policies::EpsilonGreedy>(0.1);

    auto trainer = agents::dqn::trainers::Basic{
        model,
        policy,
        optimizer,
        env_factory,
        agents::dqn::trainers::BasicOptions{}
            .batch_size_(64)
            .discount_(0.99)
            .environment_device_(torch::kCPU)
            .environment_steps_per_training_step_(4)
            .logger_(logger)
            .minimum_replay_buffer_size_(1000)
            .network_device_(torch::kCPU)
            .replay_buffer_size_(10000)
            .network_device_(torch::kCPU)
            .target_network_update_steps_(10)
    };

    trainer.run(3600);
}

Module::Module()
{
    net = register_module(
        "net",
        torch::nn::Sequential{
            torch::nn::Linear{5, 64},
            torch::nn::ReLU{true},
            torch::nn::Linear{64, 64},
            torch::nn::ReLU{true},
            torch::nn::Linear{64, 2}
        }
    );
}

std::unique_ptr<agents::dqn::modules::DQNOutput> Module::forward_impl(const torch::Tensor &states)
{
    auto values = net->forward(states);
    return std::make_unique<agents::dqn::modules::DQNOutput>(values);
}

std::unique_ptr<agents::dqn::modules::Base> Module::clone() const
{
    auto out = std::make_unique<Module>();
    auto copied_net = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(net->clone());
    assert(copied_net);
    out->net = out->replace_module("net", copied_net);
    return out;
}
