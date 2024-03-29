#include <torch/torch.h>
#include <gtest/gtest.h>

#include <rl/agents/alpha_zero/alpha_zero.h>
#include <rl/simulators/combinatorial_lock.h>


using namespace rl::agents::alpha_zero;

class ModuleOutput : public modules::BaseOutput
{    
    public:
        ModuleOutput(const torch::Tensor &values, const torch::Tensor &priors)
            : BaseOutput{priors}, values{values}
        {}

        ~ModuleOutput() = default;

        torch::Tensor value_estimates() const override { return values; }

        torch::Tensor value_loss(const torch::Tensor &rewards) const override {
            return torch::tensor(0.0f);
        }

    private:
        torch::Tensor values;
};

class Module : public modules::Base
{
    public:
        Module(int64_t dim) : dim{dim} {}

        std::unique_ptr<modules::BaseOutput> forward(const torch::Tensor &states) override {
            return std::make_unique<ModuleOutput>(
                torch::zeros({states.size(0)}),
                torch::zeros({states.size(0), dim})
            );
        }
    
    private:
        int64_t dim;
};



TEST(mcts, single_step)
{
    int sims{1000};
    int n{5};

    auto module = std::make_shared<Module>(5);
    auto sim = std::make_shared<rl::simulators::CombinatorialLock>(5, std::vector{0, 1, 2, 3, 4});

    auto states = sim->reset(n);
    auto nodes = mcts(
        states.states,
        std::dynamic_pointer_cast<rl::policies::constraints::CategoricalMask>(states.action_constraints),
        module,
        sim,
        MCTSOptions{}
            .steps_(sims)
            .dirchlet_noise_epsilon_(0.0f)
    );

    for (const auto &node : nodes) {
        auto visit_count = node->visit_count();
        ASSERT_EQ(visit_count.sum().item().toLong(), sims);
        ASSERT_EQ(visit_count.argmax().item().toLong(), 0);
    }
}
