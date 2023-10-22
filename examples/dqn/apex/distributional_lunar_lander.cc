#include <rl/rl.h>
#include <rl/remote_env/remote_env.h>
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
                    torch::nn::Linear{8, 128},
                    torch::nn::ReLU{true},
                    torch::nn::Linear{128, 128},
                    torch::nn::ReLU{true},
                    torch::nn::Linear{128, 4 * 51},
                }
            );

            atoms_ = register_buffer("atoms", torch::linspace(-100.0f, 100.0f, 51));
        }
        torch::Tensor forward(const torch::Tensor &x) override {
            return net->forward(x).view({-1, 4, 51});
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
    auto env_factory = std::make_shared<rl::remote_env::LunarLanderFactory>("localhost", 50500, 16);
    env_factory->set_logger(logger);
    auto model = std::make_shared<Module>();
    model->to(torch::kCUDA);
    auto value_parser = std::make_shared<agents::dqn::value_parsers::Distributional>(model->atoms(), true);
    auto optimizer = std::make_shared<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions{}.weight_decay(1e-5));
    auto policy = std::make_shared<agents::dqn::policies::EpsilonGreedy>(0.01);

    auto trainer = agents::dqn::trainers::Apex{
        model,
        value_parser,
        optimizer,
        policy,
        env_factory,
        agents::dqn::trainers::ApexOptions{}
            .batch_size_(64)
            .discount_(0.99)
            .double_dqn_(true)
            .workers_(2)
            .worker_batchsize_(32)
            .inference_replay_size_(10000)
            .logger_(logger)
            .minimum_replay_buffer_size_(10000)
            .n_step_(3)
            .target_network_lr_(5e-3)
            .training_buffer_size_(1000000)
            .network_device_(torch::kCUDA)
            .replay_device_(torch::kCPU)
            .environment_device_(torch::kCPU)
            .enable_inference_cuda_graph_(true)
            .enable_training_cuda_graph_(true)
    };

    trainer.run(360000);
}
