#include <rl/rl.h>
#include <torchdebug.h>


using namespace rl;


class Module : public agents::dqn::Module, public torch::nn::Cloneable<Module>
{
    public:
        Module() {
            reset();
        }
        void reset() override {
            net = register_module(
                "net",
                torch::nn::Sequential{
                    torch::nn::Linear{5, 64},
                    torch::nn::ReLU{true},
                    torch::nn::Linear{64, 64},
                    torch::nn::ReLU{true},
                    torch::nn::Linear{64, 2 * 51},
                }
            );

            atoms_ = register_buffer("atoms", torch::linspace(0.0f, 200.0f, 51));
        }
        torch::Tensor forward(const torch::Tensor &x) override {
            return net->forward(x).view({-1, 2, 51});
        }

        const torch::Tensor &atoms() const {
            return atoms_;
        }
    
    private:
        torch::nn::Sequential net;
        torch::Tensor atoms_;
};


int main()
{
    auto logger = std::make_shared<logging::client::EMA>(std::initializer_list<double>{0.6, 0.9, 0.99, 0.999}, 1);
    auto env_factory = std::make_shared<rl::env::CartPoleDiscreteFactory>(200, 2);
    env_factory->set_logger(logger);
    auto model = std::make_shared<Module>();
    model->to(torch::kCUDA);
    auto value_parser = std::make_shared<agents::dqn::value_parsers::Distributional>(model->atoms(), true);
    auto optimizer = std::make_shared<torch::optim::Adam>(model->parameters());
    auto policy = std::make_shared<agents::dqn::policies::EpsilonGreedy>(0.1);

    auto trainer = agents::dqn::trainers::Basic{
        model,
        value_parser,
        policy,
        optimizer,
        env_factory,
        agents::dqn::trainers::BasicOptions{}
            .batch_size_(64)
            .discount_(0.99)
            .environment_device_(torch::kCPU)
            .environment_steps_per_training_step_(8.0f)
            .logger_(logger)
            .minimum_replay_buffer_size_(10000)
            .network_device_(torch::kCUDA)
            .replay_buffer_size_(100000)
            .replay_device_(torch::kCPU)
            .target_network_lr_(1e-3)
            .n_step_(3)
            .enable_cuda_graph_(true)
    };

    trainer.run(3600);
}
