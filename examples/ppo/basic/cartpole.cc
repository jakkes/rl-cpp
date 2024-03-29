#include <initializer_list>

#include <argparse/argparse.hpp>
#include <rl/rl.h>

#include <torchdebug.h>


using namespace rl;

argparse::ArgumentParser parse_args(int argc, char **argv)
{
    argparse::ArgumentParser parser{};
    parser
        .add_argument("--duration")
        .help("Train duration, seconds.")
        .default_value(60)
        .scan<'i', int>();
    
    parser
        .add_argument("--envs")
        .help("Number of environment sequences handled in parallell.")
        .default_value<int>(32)
        .scan<'i', int>();

    parser
        .add_argument("--env-worker-threads")
        .help("Environment rollouts are parallellized across threads.")
        .default_value<int>(4)
        .scan<'i', int>();

    parser
        .add_argument("--actions")
        .help("Number of actions the action space is discretized into.")
        .default_value<int>(2)
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

        Model(int actions) :
            base{register_module("base", torch::nn::Linear{5, 64})},
            value{register_module("value", torch::nn::Linear{64, 1})},
            policy{register_module("policy", torch::nn::Linear{64, actions})}
        {}

        std::unique_ptr<rl::agents::ppo::ModuleOutput> forward(
                                                    const torch::Tensor &input)
        {
            auto re = std::make_unique<rl::agents::ppo::ModuleOutput>();
            auto base = torch::relu(this->base->forward(input));
            re->value = value->forward(base).squeeze(-1);
            re->policy = std::make_unique<policies::Categorical>(torch::softmax(policy->forward(base), -1));
            return re;
        }

};

int main(int argc, char **argv)
{
    auto args = parse_args(argc, argv);

    auto model = std::make_shared<Model>(args.get<int>("--actions"));
    auto logger = std::make_shared<logging::client::EMA>(std::initializer_list<double>{0.0, 0.6, 0.9, 0.99, 0.999, 0.9999}, 5);
    auto env_factory = std::make_shared<env::CartPoleDiscreteFactory>(200, args.get<int>("--actions"));
    env_factory->set_logger(logger);
    auto optimizer = std::make_shared<torch::optim::Adam>(model->parameters());

    agents::ppo::trainers::Basic trainer{
        model,
        optimizer,
        env_factory,
        agents::ppo::trainers::BasicOptions{}
            .cuda_(false)
            .env_workers_(args.get<int>("--env-worker-threads"))
            .envs_(args.get<int>("--envs"))
            .logger_(logger)
    };

    trainer.run(std::chrono::seconds(args.get<int>("--duration")));
}
