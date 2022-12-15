#include <rl/rl.h>
#include <torch/torch.h>
#include <argparse/argparse.hpp>

#include "torchdebug.h"

using namespace rl;
using namespace torch;
using namespace torch::indexing;

class Net : public agents::alpha_zero::modules::Base
{
    public:
        Net(int dim, int length);
        
        std::unique_ptr<agents::alpha_zero::modules::BaseOutput> forward(
                                            const torch::Tensor &states) override;

    private:
        const int dim, length;
        nn::Sequential policy, value;
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
        .default_value<int>(20)
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
    auto net = std::make_shared<Net>( dim, length );
    auto optimizer = std::make_shared<optim::Adam>(net->parameters());
    auto temperature_control = std::make_shared<rl::utils::float_control::TimedExponentialDecay>(1.0f, 0.1f, 300);

    std::vector<int> correct_sequence{};
    for (int i = 0; i < args.get<int>("--length"); i++) {
        correct_sequence.push_back(i % dim);
    }
    auto sim = std::make_shared<simulators::CombinatorialLock>(
        dim,
        correct_sequence,
        simulators::CombinatorialLockOptions{}.intermediate_rewards_(true)
    );

    agents::alpha_zero::Trainer trainer {
        net,
        optimizer,
        sim,
        agents::alpha_zero::TrainerOptions{}
            .logger_(logger)
            .max_episode_length_(length)
            .min_replay_size_(100)
            .replay_size_(10000)
            .module_device_(torch::kCPU)
            .self_play_batchsize_(1)
            .self_play_mcts_steps_(length)
            .self_play_dirchlet_noise_alpha_(0.1)
            .self_play_dirchlet_noise_epsilon_(0.25)
            .self_play_temperature_control_(temperature_control)
            .self_play_workers_(1)
            .training_batchsize_(64)
            .training_mcts_steps_(10 * length)
            .training_dirchlet_noise_alpha_(0.25)
            .training_dirchlet_noise_epsilon_(0.1)
            .training_temperature_control_(temperature_control)
    };

    trainer.run(3600);
}


Net::Net(int dim, int length)
: dim{dim}, length{length}
{
    policy = register_module(
        "policy",
        nn::Sequential{
            nn::Linear{dim+2, 32},
            nn::ReLU{nn::ReLUOptions{true}},
            nn::Linear{32, dim}
        }
    );

    value = register_module(
        "value",
        nn::Sequential{
            nn::Linear{dim+2, 32},
            nn::ReLU{nn::ReLUOptions{true}},
            nn::Linear{32, 32},
            nn::ReLU{nn::ReLUOptions{true}},
            nn::Linear{32, 32},
            nn::ReLU{nn::ReLUOptions{true}},
            nn::Linear{32, 1}
        }
    );
}

std::unique_ptr<agents::alpha_zero::modules::BaseOutput> Net::forward(const torch::Tensor &states)
{
    auto batchsize = states.size(0);
    auto one_hot_encoded = torch::zeros({batchsize, dim+2});

    auto length = (states >= 0).sum(1);
    one_hot_encoded.index_put_({torch::arange(batchsize), length.remainder(dim)}, 1.0f);

    auto correct = torch::ones({batchsize}, torch::TensorOptions{}.dtype(torch::kBool));

    auto state_accessor = states.accessor<int64_t, 2>();
    auto one_hot_encoded_accessor = one_hot_encoded.accessor<float, 2>();
    auto correct_accessor = correct.accessor<bool, 1>();

    for (int64_t i = 0; i < batchsize; i++) {
        for (int j = 0; j < dim; j++) {
            auto value = state_accessor[i][j] % dim;
            if (value < 0) {
                break;
            }

            if (value != j) {
                correct_accessor[i] = false;
                break;
            }
        }
    }

    for (int64_t i = 0; i < batchsize; i++) {
        if (correct_accessor[i]) {
            one_hot_encoded_accessor[i][dim + 1] = 1.0f;
        }
    }
    
    auto policy_logits = policy->forward(one_hot_encoded);
    auto values = value->forward(one_hot_encoded).squeeze(1);

    return std::make_unique<agents::alpha_zero::modules::MeanValueOutput>(
        policy_logits, values
    );
}
