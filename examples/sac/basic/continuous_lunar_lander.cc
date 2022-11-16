#include <rl/rl.h>
#include <torchdebug.h>

using namespace rl;
using namespace torch::indexing;

class Actor : public rl::agents::sac::Actor
{
    public:
        torch::nn::Sequential policy, value;

        Actor() {
            policy = register_module(
                "policy",
                torch::nn::Sequential{
                    torch::nn::Linear{8, 128},
                    torch::nn::ReLU{true},
                    torch::nn::Linear{128, 128},
                    torch::nn::ReLU{true},
                    torch::nn::Linear{128, 4}
                }
            );
            value = register_module(
                "value",
                torch::nn::Sequential{
                    torch::nn::Linear{8, 128},
                    torch::nn::ReLU{true},
                    torch::nn::Linear{128, 128},
                    torch::nn::ReLU{true},
                    torch::nn::Linear{128, 1}
                }
            );
        }

        rl::agents::sac::ActorOutput forward(
                                    const torch::Tensor &states) override
        {
            auto policy_output = policy->forward(states);
            auto mean = policy_output.index({"...", Slice(0, 2)});
            auto std = 1.0f + torch::elu(policy_output.index({"...", Slice(2, 4)}));
            auto value_output = value->forward(states).squeeze(-1);

            return rl::agents::sac::ActorOutput{mean, std, value_output};
        }

        std::unique_ptr<rl::agents::sac::Actor> clone() const
        {
            auto out = std::make_unique<Actor>();
            out->policy = out->replace_module("policy", std::dynamic_pointer_cast<torch::nn::SequentialImpl>(policy->clone()));
            out->value = out->replace_module("value", std::dynamic_pointer_cast<torch::nn::SequentialImpl>(value->clone()));
            return out;
        }
};


class Critic : public rl::agents::sac::Critic
{
    public:
        torch::nn::Sequential Q;

    public:
        Critic() {
            Q = register_module(
                "Q",
                torch::nn::Sequential{
                    torch::nn::Linear{10, 128},
                    torch::nn::ReLU{true},
                    torch::nn::Linear{128, 128},
                    torch::nn::ReLU{true},
                    torch::nn::Linear{128, 1}
                }
            );
        };

        rl::agents::sac::CriticOutput forward(
            const torch::Tensor &states,
            const torch::Tensor &actions
        )
        {
            return rl::agents::sac::CriticOutput{
                Q->forward(torch::concat({states, actions}, -1)).squeeze(-1)
            };
        }

        std::unique_ptr<rl::agents::sac::Critic> clone() const
        {
            auto out = std::make_unique<Critic>();
            out->Q = out->replace_module("Q", std::dynamic_pointer_cast<torch::nn::SequentialImpl>(Q->clone()));
            return out;
        }
};


int main()
{
    auto actor = std::make_shared<Actor>();
    actor->to(torch::kCUDA);
    std::vector<std::shared_ptr<agents::sac::Critic>> critics{ 
        std::make_shared<Critic>(), 
        std::make_shared<Critic>()
    };
    critics[0]->to(torch::kCUDA);
    critics[1]->to(torch::kCUDA);

    auto logger = std::make_shared<logging::client::EMA>(
        std::initializer_list<double>{0.0, 0.6, 0.9, 0.99, 0.999, 0.9999},
        5
    );
    auto env_factory = std::make_shared<env::remote::ContinuousLunarLanderFactory>("localhost:50051");
    env_factory->set_logger(logger);
    
    auto actor_optimizer = std::make_shared<torch::optim::Adam>(
        actor->parameters(),
        torch::optim::AdamOptions{}.weight_decay(1e-6)
    );
    std::vector<std::shared_ptr<torch::optim::Optimizer>> critic_optimizers {
        std::make_shared<torch::optim::Adam>(
            critics[0]->parameters(),
            torch::optim::AdamOptions{}.weight_decay(1e-6)
        ),
        std::make_shared<torch::optim::Adam>(
            critics[1]->parameters(),
            torch::optim::AdamOptions{}.weight_decay(1e-6)
        )
    };

    agents::sac::trainers::Basic trainer {
        actor,
        critics,
        actor_optimizer,
        critic_optimizers,
        env_factory,
        agents::sac::trainers::BasicOptions{}
            .action_range_max_(1.0f)
            .action_range_min_(-1.0f)
            .batch_size_(64)
            .discount_(0.99f)
            .environment_steps_per_training_step_(2.0f)
            .logger_(logger)
            .minimum_replay_buffer_size_(1000)
            .replay_buffer_size_(100000)
            .target_network_lr_(5e-3)
            .temperature_(0.001f)
            .replay_device_(torch::kCPU)
            .network_device_(torch::kCUDA)
            .environment_device_(torch::kCPU)
            .huber_loss_delta_(10.0f)
    };

    trainer.run(3600);
}
