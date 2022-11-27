#ifndef RL_AGENTS_ALPHA_ZERO_MCTS_H_
#define RL_AGENTS_ALPHA_ZERO_MCTS_H_


#include <memory>
#include <vector>

#include <torch/torch.h>

#include <rl/policies/constraints/categorical_mask.h>
#include <rl/option.h>

#include "modules/base.h"


namespace rl::agents::alpha_zero
{
    struct MCTSOptions
    {
        RL_OPTION(torch::Device, module_device) = torch::kCPU;

        RL_OPTION(float, c1) = 1.25f;
        RL_OPTION(float, c2) = 19652;
    };

    class MCTSNode;
    struct MCTSSelectResult
    {
        std::shared_ptr<MCTSNode> node;
        int64_t action;
    };

    class MCTSNode
    {
        public:
            MCTSNode(
                const torch::Tensor &state,
                const torch::Tensor &prior
            );

            inline
            std::shared_ptr<MCTSNode> get_child(int i) { return children[i]; }

            MCTSSelectResult select(const MCTSOptions &options={}) const;

        private:
            int64_t dim;
            torch::Tensor state, P, Q, N;
            torch::TensorAccessor<float, 1> *P_accessor, *Q_accessor;
            torch::TensorAccessor<int64_t, 1> *N_accessor;
            std::vector<std::shared_ptr<MCTSNode>> children;
    };

    void mcts(
        std::vector<std::shared_ptr<MCTSNode>> *root_nodes,
        std::shared_ptr<modules::Base> module,
        const MCTSOptions &options={}
    );

    std::vector<std::shared_ptr<MCTSNode>> mcts(
        const torch::Tensor &states,
        const std::shared_ptr<rl::policies::constraints::CategoricalMask> masks,
        std::shared_ptr<modules::Base> module,
        const MCTSOptions &options={}
    );
}

#endif /* RL_AGENTS_ALPHA_ZERO_MCTS_H_ */
