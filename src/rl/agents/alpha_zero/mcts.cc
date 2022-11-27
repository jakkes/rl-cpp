#include "rl/agents/alpha_zero/mcts.h"


namespace rl::agents::alpha_zero
{

    MCTSNode::MCTSNode(
        const torch::Tensor &state,
        const torch::Tensor &prior
    ) : state{state}
    {
        dim = prior.size(0);

        this->prior = prior.to(torch::kCPU);
        *prior_accessor = this->prior.accessor<float, 1>();
        
        Q = torch::zeros_like(this->prior);
        *Q_accessor = Q.accessor<float, 1>();

        static auto N_options = torch::TensorOptions{}.dtype(torch::kLong);
        N = torch::zeros({Q.size(0)}, N_options);
        *N_accessor = N.accessor<int64_t, 1>();

        children.resize(dim);
    }

    void mcts(
        std::vector<std::shared_ptr<MCTSNode>> *root_nodes_,
        std::shared_ptr<modules::Base> module,
        const MCTSOptions &options
    )
    {
        auto &root_nodes{*root_nodes_};
    }

    std::vector<std::shared_ptr<MCTSNode>> mcts(
        const torch::Tensor &states,
        std::shared_ptr<rl::policies::constraints::CategoricalMask> masks,
        std::shared_ptr<modules::Base> module,
        const MCTSOptions &options
    )
    {
        auto output = module->forward(states.to(options.module_device));
        auto prior_policy = output->policy();
        prior_policy.include(masks);

        auto priors = prior_policy.get_probabilities();

        std::vector<std::shared_ptr<MCTSNode>> root_nodes{};
        root_nodes.resize(priors.size(0));

        for (int i = 0; i < root_nodes.size(); i++) {
            root_nodes[i] = std::make_shared<MCTSNode>(states.index({i}), priors.index({i}));
        }

    }
}
