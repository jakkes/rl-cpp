#include <argparse/argparse.hpp>
#include <rl/rl.h>

#include <torchdebug.h>


using namespace rl;

argparse::ArgumentParser parse_args(int argc, char **argv)
{
    argparse::ArgumentParser args{};
    
    args
        .add_argument("--cuda")
        .help("If set, training happens on GPU.")
        .default_value<bool>(true)
        .implicit_value(true);

    args
        .add_argument("--batch-size")
        .help("Batch size used in training.")
        .default_value<int>(64)
        .scan<'i', int>();

    args
        .add_argument("--env-workers")
        .help("Number of threads executing environment steps.")
        .default_value<int>(4)
        .scan<'i', int>();
    
    args
        .add_argument("--envs-per-worker")
        .help("Number of environments executed per environment worker.")
        .default_value<int>(4)
        .scan<'i', int>();
    
    args
        .add_argument("--inference-batch-size")
        .help("Maximum batch size per inference request.")
        .default_value<int>(16)
        .scan<'i', int>();
    
    args
        .add_argument("--inference-delay")
        .help("Maximum delay (milliseconds) before inference requests are processed.")
        .default_value<int>(100)
        .scan<'i', int>();

    args
        .add_argument("--inference-replay-size")
        .help("Number of sequences collected before added (batched) into the training replay.")
        .default_value<int>(500)
        .scan<'i', int>();
    
    args
        .add_argument("--min-replay-size")
        .help("Size of replay when training may start.")
        .default_value<int>(1000)
        .scan<'i', int>();

    args
        .add_argument("--replay-size")
        .help("Size of training replay buffer.")
        .default_value<int>(1000)
        .scan<'i', int>();
    
    args
        .add_argument("--sequence-length")
        .help("Roll out sequence lengths.")
        .default_value<int>(64)
        .scan<'i', int>();
    
    args
        .add_argument("--max-update-frequency")
        .help("Maximum number of training steps executed per second.")
        .default_value<float>(25.0f)
        .scan<'g', float>();
    
    args
        .add_argument("--duration")
        .help("Training duration (seconds).")
        .default_value<int>(3600)
        .scan<'i', int>();

    try {
        args.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << args;
        std::exit(1);
    }

    return args;
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
    auto logger = std::make_shared<logging::client::Tensorboard>();
    auto env_factory = std::make_shared<env::CartPoleContinuousFactory>(200);
    env_factory->set_logger(logger);

    if (args.get<bool>("--cuda")) {
        model->to(torch::kCUDA);
    }
    auto optimizer = std::make_shared<torch::optim::Adam>(model->parameters());

    agents::ppo::trainers::SEED trainer{
        model,
        optimizer,
        env_factory,
        agents::ppo::trainers::SEEDOptions{}
            .batchsize_(args.get<int>("--batch-size"))
            .env_workers_(args.get<int>("--env-workers"))
            .envs_per_worker_(args.get<int>("--envs-per-worker"))
            .inference_batchsize_(args.get<int>("--inference-batch-size"))
            .inference_max_delay_ms_(args.get<int>("--inference-delay"))
            .inference_replay_size_(args.get<int>("--inference-replay-size"))
            .min_replay_size_(args.get<int>("--min-replay-size"))
            .replay_size_(args.get<int>("--replay-size"))
            .sequence_length_(args.get<int>("--sequence-length"))
            .max_update_frequency_(args.get<float>("--max-update-frequency"))
            .logger_(logger)
            .replay_device_(args.get<bool>("--cuda") ? torch::kCUDA : torch::kCPU)
            .network_device_(args.get<bool>("--cuda") ? torch::kCUDA : torch::kCPU)
            .environment_device_(torch::kCPU)
    };

    trainer.run(std::chrono::seconds(args.get<int>("--duration")));
}
