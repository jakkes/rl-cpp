#include <torch/torch.h>
#include <gtest/gtest.h>

#include <rl/agents/alpha_zero/alpha_zero.h>



using namespace rl::agents::alpha_zero;

class ModuleOutput : public modules::BaseOutput
{    
    public:
        ModuleOutput(const torch::Tensor &values, const torch::Tensor &priors)
            : values{values}, priors{priors}
        {}

        ~ModuleOutput() = default;

        const rl::policies::Categorical &policy() const override { return priors; }

        torch::Tensor value_estimates() const override { return values; }

    private:
        torch::Tensor values;
        rl::policies::Categorical priors;
};

class Module : public modules::Base
{
    public:
        Module(int64_t dim) : dim{dim} {}

        std::unique_ptr<modules::BaseOutput> forward(const torch::Tensor &states) override {
            return std::make_unique<ModuleOutput>(
                torch::randn({states.size(0)}),
                torch::randn({states.size(0), dim})
            );
        }
    
    private:
        int64_t dim;
};

class Simulator : rl::simulators::Base
{
    public:
        Simulator(int64_t dim, int64_t length) : dim{dim}, length{length} {
            correct_sequence = torch::arange(length).remainder(dim);
        }
        
        rl::simulators::States reset(int64_t n) const override
        {
            rl::simulators::States out{};
            out.states = torch::zeros({n, length, dim}));
            out.action_constraints = std::make_shared<rl::policies::constraints::CategoricalMask>(
                torch::ones({n, dim}, torch::TensorOptions{}.dtype(torch::kBool))
            );
            return out;
        }

        rl::simulators::Observations step(const torch::Tensor &states, const torch::Tensor &actions) const override
        {
            rl::simulators::Observations out{};
            auto sequence_lengths = (states > 1.0f).any(-1).sum(-1);
            auto batchvec = torch::arange(states.size(0));

            out.next_states = states.index_put({batchvec, sequence_lengths, actions}, torch::ones({states.size(0)}));
            out.
        }

    private:
        int64_t dim, length;
        torch::Tensor correct_sequence;
};

TEST(mcts, run)
{

}
