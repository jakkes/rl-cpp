#include <initializer_list>

#include <argparse/argparse.hpp>
#include <rl/rl.h>
#include <torch/torch.h>

#include <torchdebug.h>


using namespace rl;
using namespace torch::indexing;

argparse::ArgumentParser parse_args(int argc, char **argv)
{
    argparse::ArgumentParser parser{};
    parser
        .add_argument("--duration")
        .help("Train duration, seconds.")
        .default_value(3600)
        .scan<'i', int>();
    
    parser
        .add_argument("--envs")
        .help("Number of environment sequences handled in paralell.")
        .default_value<int>(32)
        .scan<'i', int>();

    parser
        .add_argument("--env-worker-threads")
        .help("Environment rollouts are parallellized across threads.")
        .default_value<int>(4)
        .scan<'i', int>();

    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }

    return parser;
}

class Model : public rl::agents::ppo::Module
{

    public:

        torch::nn::Linear base;
        torch::nn::Linear value;
        torch::nn::Linear policy;

        Model() :
            base{register_module("base", torch::nn::Linear{5, 64})},
            value{register_module("value", torch::nn::Linear{64, 1})},
            policy{register_module("policy", torch::nn::Linear{64, 2})}
        {}

        std::unique_ptr<rl::agents::ppo::ModuleOutput> forward(
                                                    const torch::Tensor &input)
        {
            auto re = std::make_unique<rl::agents::ppo::ModuleOutput>();
            auto base = torch::relu(this->base->forward(input));
            re->value = value->forward(base).squeeze(-1);
            auto policy_output = policy->forward(base);
            policy_output = torch::elu(policy_output) + 1;
            
            re->policy = std::make_unique<policies::Beta>(
                policy_output.index({"...", 0}),
                policy_output.index({"...", 1}),
                - torch::ones_like(policy_output.index({"...", 1})),
                torch::ones_like(policy_output.index({"...", 1}))
            );
            return re;
        }

};

int main(int argc, char **argv)
{
    auto args = parse_args(argc, argv);

    auto model = std::make_shared<Model>();
    auto logger = std::make_shared<logging::client::EMA>(std::initializer_list<double>{0.0, 0.6, 0.9, 0.99, 0.999, 0.9999}, 5);
    auto env_factory = std::make_shared<env::CartPoleContinuousFactory>(200, logger);
    
    if (torch::cuda::is_available()) {
        env_factory->cuda();
        model->to(torch::kCUDA);
    }
    auto optimizer = std::make_shared<torch::optim::Adam>(model->parameters());

    agents::ppo::trainers::Basic trainer{
        model,
        optimizer,
        env_factory,
        agents::ppo::trainers::BasicOptions{}
            .cuda_(torch::cuda::is_available())
            .env_workers_(args.get<int>("--env-worker-threads"))
            .envs_(args.get<int>("--envs"))
            .logger_(logger)
    };

    trainer.run(std::chrono::seconds(args.get<int>("--duration")));
}
