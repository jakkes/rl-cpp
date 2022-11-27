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
    class MCTSNode
    {
        public:
            MCTSNode(
                const torch::Tensor &state,
                const torch::Tensor &prior
            );

            inline
            std::shared_ptr<MCTSNode> get_child(int i) { return children[i]; }

        private:
            int64_t dim;
            torch::Tensor state, prior, Q, N;
            torch::TensorAccessor<float, 1> *prior_accessor, *Q_accessor;
            torch::TensorAccessor<int64_t, 1> *N_accessor;
            std::vector<std::shared_ptr<MCTSNode>> children;
    };

    struct MCTSOptions
    {
        RL_OPTION(torch::Device, module_device) = torch::kCPU;
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
