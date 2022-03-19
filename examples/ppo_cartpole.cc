#include <argparse/argparse.hpp>
#include <rl/rl.h>


using namespace rl;

argparse::ArgumentParser parse_args(int argc, char **argv)
{
    argparse::ArgumentParser parser{};
    parser
        .add_argument("--duration")
        .help("Train duration, seconds.")
        .default_value(60)
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
            base{register_module("base", torch::nn::Linear{4, 64})},
            value{register_module("value", torch::nn::Linear{64, 1})},
            policy{register_module("policy", torch::nn::Linear{64, 2})}
        {}

        std::unique_ptr<rl::agents::ppo::ModuleOutput> forward(
                                                    const torch::Tensor &input)
        {
            auto re = std::make_unique<rl::agents::ppo::ModuleOutput>();
            auto base = torch::relu(this->base->forward(input));
            re->value = value->forward(base).squeeze_(-1);
            re->policy = std::make_unique<policies::Categorical>(torch::softmax(policy->forward(base), -1));
            return re;
        }

};

int main(int argc, char **argv)
{

    auto args = parse_args(argc, argv);

    auto model = std::make_shared<Model>();
    auto env_factory = std::make_shared<env::CartPoleFactory>(200);
    
    if (torch::cuda::is_available())
    {
        env_factory->cuda();
        model->to(torch::kCUDA);
    }
    auto optimizer = std::make_shared<torch::optim::Adam>(model->parameters());

    agents::ppo::trainers::Basic trainer{
        model,
        optimizer,
        env_factory
    };

    trainer.run(std::chrono::seconds(args.get<int>("--duration")));
}
