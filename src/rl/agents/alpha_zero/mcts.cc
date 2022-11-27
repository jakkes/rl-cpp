#include "rl/agents/alpha_zero/mcts.h"


namespace rl::agents::alpha_zero
{

    MCTSNode::MCTSNode(
        const torch::Tensor &state,
        const torch::Tensor &prior
    ) : state{state}
    {
        dim = prior.size(0);

        this->P = prior.to(torch::kCPU);
        *P_accessor = this->P.accessor<float, 1>();
        
        Q = torch::zeros_like(this->P);
        *Q_accessor = Q.accessor<float, 1>();

        static auto N_options = torch::TensorOptions{}.dtype(torch::kLong);
        N = torch::zeros({Q.size(0)}, N_options);
        *N_accessor = N.accessor<int64_t, 1>();

        children.resize(dim);
    }

    MCTSSelectResult MCTSNode::select(const MCTSOptions &options) const
    {
        auto a = torch::argmax(
            Q
            + P * N.sum().sqrt_() / (1 + N) 
                * (options.c1 + ((N.sum() + options.c2 + 1.0f) / options.c2).log_())
        ).item().toLong();

        if (children[a]) {
            return children[a]->select(options);
        }
        else
        {
            MCTSSelectResult out{};
        }
    }

    static inline
    std::shared_ptr<MCTSNode> selection(std::shared_ptr<MCTSNode> node)
    {

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
