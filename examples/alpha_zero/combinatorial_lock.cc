#include <rl/rl.h>
#include <torch/torch.h>
#include <argparse/argparse.hpp>

#include "torchdebug.h"

using namespace rl;
using namespace torch;

class Net : public agents::alpha_zero::modules::Base
{
    public:
        Net(int dim, int length, int atoms);
        
        std::unique_ptr<agents::alpha_zero::modules::BaseOutput> forward(
                                            const torch::Tensor &states) override;

    private:
        const int dim, length, atoms;
        nn::Sequential fc1, policy, value;
};

argparse::ArgumentParser parse_args(int argc, char **argv)
{
    argparse::ArgumentParser parser{};

    parser
        .add_argument("--dim")
        .default_value<int>(10)
        .scan<'i', int>();

    parser
        .add_argument("--length")
        .default_value<int>(10)
        .scan<'i', int>();

    try {
        parser.parse_args(argc, argv);
        return parser;
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }
}


int main(int argc, char **argv)
{
    auto args = parse_args(argc, argv);
    int length = args.get<int>("--length");
    int dim = args.get<int>("--dim");

    auto logger = std::make_shared<logging::client::EMA>(std::vector{0.6, 0.9, 0.99}, 10, 10);
    auto net = std::make_shared<Net>( dim, length, 11);
    auto optimizer = std::make_shared<optim::Adam>(net->parameters());

    std::vector<int> correct_sequence{};
    for (int i = 0; i < args.get<int>("--length"); i++) {
        correct_sequence.push_back(i % dim);
    }
    auto sim = std::make_shared<simulators::CombinatorialLock>(
        dim,
        correct_sequence
    );

    agents::alpha_zero::Trainer trainer {
        net,
        optimizer,
        sim,
        agents::alpha_zero::TrainerOptions{}
            .logger_(logger)
            .max_episode_length_(length)
            .min_replay_size_(1000)
            .replay_size_(100000)
            .module_device_(torch::kCPU)
            .self_play_batchsize_(64)
            .self_play_mcts_options_(
                agents::alpha_zero::MCTSOptions{}
                    .steps_(length / 4)
            )
            .self_play_temperature_(1.0f)
            .self_play_workers_(1)
            .training_batchsize_(128)
            .training_mcts_options_(
                agents::alpha_zero::MCTSOptions{}
                    .steps_(length / 2)
            )
            .training_temperature_(0.1)
    };

    trainer.run(3600);
}


Net::Net(int dim, int length, int atoms)
: dim{dim}, length{length}, atoms{atoms}
{
    fc1 = register_module(
        "fc1",
        nn::Sequential{
            nn::Linear{dim+1, 64},
            nn::ReLU{true},
            nn::Linear{64, 32}
        }
    );

    policy = register_module(
        "policy",
        nn::Sequential{
            nn::Linear{length * 32, 512},
            nn::ReLU{true},
            nn::Linear{512, 256},
            nn::ReLU{true},
            nn::Linear{256, dim}
        }
    );

    value = register_module(
        "value",
        nn::Sequential{
            nn::Linear{length * 32, 512},
            nn::ReLU{true},
            nn::Linear{512, 256},
            nn::ReLU{true},
            nn::Linear{256, atoms}
        }
    );
}

std::unique_ptr<agents::alpha_zero::modules::BaseOutput> Net::forward(const torch::Tensor &states)
{
    auto batchsize = states.size(0);
    auto one_hot_encoded = torch::zeros({batchsize, length, dim+1});
    one_hot_encoded.index_put_(
        {
            torch::arange(batchsize).repeat_interleave(length),
            torch::arange(length).repeat(batchsize),
            states.reshape({-1})
        },
        1.0f
    );

    auto encoding = fc1->forward(one_hot_encoded.reshape({-1, dim+1}));
    auto policy_logits = policy->forward(encoding.reshape({batchsize, -1}));
    auto value_logits = value->forward(encoding.reshape({batchsize, -1}));

    return std::make_unique<agents::alpha_zero::modules::FixedValueSupportOutput>(
        policy_logits, value_logits, 0.0f, 1.0f, atoms
    );
}
