#include <torch/torch.h>
#include <gtest/gtest.h>

#include <rl/agents/alpha_zero/alpha_zero.h>
#include <rl/simulators/combinatorial_lock.h>


using namespace rl::agents::alpha_zero;

class FastMCTSTestModuleOutput : public modules::BaseOutput
{    
    public:
        FastMCTSTestModuleOutput(const torch::Tensor &values, const torch::Tensor &priors)
            : BaseOutput{priors}, values{values}
        {}

        ~FastMCTSTestModuleOutput() = default;

        torch::Tensor value_estimates() const override { return values; }

        torch::Tensor value_loss(const torch::Tensor &rewards) const override {
            return torch::tensor(0.0f);
        }

    private:
        torch::Tensor values;
};

class FastMCTSTestModule : public modules::Base
{
    public:
        FastMCTSTestModule(int64_t dim) : dim{dim} {}

        std::unique_ptr<modules::BaseOutput> forward(const torch::Tensor &states) override {
            return std::make_unique<FastMCTSTestModuleOutput>(
                torch::zeros({states.size(0)}, torch::TensorOptions{}.device(states.device())),
                torch::zeros({states.size(0), dim}, torch::TensorOptions{}.device(states.device()))
            );
        }
    
    private:
        int64_t dim;
};


TEST(fast_mcts, simple_sim)
{
    int sims{10000};
    int n{5};

    auto device = torch::kCPU;

    auto module = std::make_shared<FastMCTSTestModule>(5);
    module->to(device);
    auto combination = std::vector{3, 1, 2, 4, 0};
    auto sim = std::make_shared<rl::simulators::CombinatorialLock>(5, combination, rl::simulators::CombinatorialLockOptions{}.device_(device));

    auto states = sim->reset(n);

    auto inference_fn = [&] (const torch::Tensor &states) {
        FastMCTSInferenceResult out{};
        auto module_output = module->forward(states);
        out.probabilities = module_output->policy().get_probabilities();
        out.values = module_output->value_estimates();
        return out;
    };

    FastMCTSExecutorOptions options{};
    options.steps_(sims);
    options.dirchlet_noise_epsilon_(0.0f);
    options.sim_device_(device);
    options.module_device_(device);

    FastMCTSExecutor mcts{
        states.states,
        std::dynamic_pointer_cast<rl::policies::constraints::CategoricalMask>(states.action_constraints)->mask(),
        inference_fn,
        sim,
        options
    };

    for (int j = 0; j < 5; j++) {

        mcts.run();
        auto visit_counts = mcts.current_visit_counts();
        for (int i = 0; i < visit_counts.size(0); i++) {
            auto visit_count = visit_counts.index({i});
            ASSERT_GE(visit_count.sum().item().toLong(), sims);
        }

        ASSERT_TRUE( (visit_counts.argmax(1) == combination[j]).all().item().toBool() );
        auto actions = visit_counts.argmax(1);
        mcts.step(actions);

        if (j < 4) {
            ASSERT_TRUE( (mcts.current_visit_counts().sum(1) > 0).all().item().toBool() );
        }
    }
}
