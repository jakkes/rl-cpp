#include <rl/rl.h>
#include <torch/torch.h>

#include "torchdebug.h"


using namespace rl;
using namespace torch;
using namespace torch::indexing;


class Net : public agents::alpha_zero::modules::Base
{
    public:
        Net();

        std::unique_ptr<agents::alpha_zero::modules::BaseOutput> forward(
            const Tensor &states) override;
    private:
        nn::Sequential policy, value;
};


Net::Net()
{
    policy = register_module(
        "policy",
        nn::Sequential{
            nn::Linear{5, 64},
            nn::ReLU{nn::ReLUOptions{true}},
            nn::Linear{64, 2}
        }
    );

    value = register_module(
        "value",
        nn::Sequential{
            nn::Linear{5, 64},
            nn::ReLU{nn::ReLUOptions{true}},
            nn::Linear{64, 11}
        }
    );
}


int main()
{
    auto logger = std::make_shared<logging::client::Tensorboard>(logging::client::TensorboardOptions{}.frequency_window_(10));
    auto net = std::make_shared<Net>();
    net->to(torch::kCUDA);
    auto optimizer = std::make_shared<optim::Adam>(net->parameters());
    auto temperature_control = std::make_shared<rl::utils::float_control::TimedExponentialDecay>(1.0, 0.5, 600);

    auto sim = std::make_shared<simulators::DiscreteCartPole>(
        200, 2, simulators::CartPoleOptions{}.reward_scaling_factor_(1.0f / 200).sparse_reward_(true)
    );

    agents::alpha_zero::Trainer trainer {
        net,
        optimizer,
        sim,
        agents::alpha_zero::TrainerOptions{}
            .logger_(logger)
            .max_episode_length_(200)
            .min_replay_size_(1000)
            .replay_size_(100000)
            .module_device_(torch::kCUDA)
            .self_play_batchsize_(32)
            .self_play_mcts_steps_(80)
            .self_play_dirchlet_noise_alpha_(0.5)
            .self_play_dirchlet_noise_epsilon_(0.25)
            .self_play_temperature_control_(temperature_control)
            .self_play_workers_(8)
            .training_batchsize_(64)
            .training_mcts_steps_(80)
            .training_dirchlet_noise_alpha_(0.5)
            .training_dirchlet_noise_epsilon_(0.25)
            .training_temperature_control_(temperature_control)
            .training_workers_(8)
            .discount_(1.0)
    };

    trainer.run(10800);
}


std::unique_ptr<agents::alpha_zero::modules::BaseOutput> Net::forward(const Tensor &states_)
{
    auto states = states_.clone();
    states.index({Slice(), 4}).sub_(100.0f).div_(100.0f);

    auto policy_logits = policy->forward(states);
    auto value_logits = value->forward(states);

    return std::make_unique<agents::alpha_zero::modules::FixedValueSupportOutput>(
        policy_logits, value_logits, 0.0f, 1.0f, 11
    );
}
