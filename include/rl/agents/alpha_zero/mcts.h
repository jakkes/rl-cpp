#ifndef RL_AGENTS_ALPHA_ZERO_MCTS_H_
#define RL_AGENTS_ALPHA_ZERO_MCTS_H_


#include <memory>
#include <vector>

#include <torch/torch.h>

#include <rl/simulators/base.h>
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

        RL_OPTION(float, discount) = 1.0f;
        RL_OPTION(int, steps) = 100;

        RL_OPTION(float, dirchlet_noise_alpha) = 0.1f;
        RL_OPTION(float, dirchlet_noise_epsilon) = 0.5f;
    };

    class MCTSNode;
    struct MCTSSelectResult
    {
        MCTSNode *node;
        int64_t action;
    };

    class MCTSNode
    {
        public:
            MCTSNode(
                const torch::Tensor &state,
                const torch::Tensor &mask,
                const torch::Tensor &prior,
                float value
            );

            inline
            std::shared_ptr<MCTSNode> get_child(int i) const { return children[i]; }

            inline
            const torch::Tensor state() const { return state_; }

            inline
            const torch::Tensor mask() const { return mask_; }

            inline
            const torch::Tensor visit_count() const { return N; }

            MCTSSelectResult select(const MCTSOptions &options={});

            void expand(
                int64_t action,
                float reward,
                bool terminal,
                const torch::Tensor &next_state,
                const torch::Tensor &next_mask,
                const torch::Tensor &next_prior,
                const torch::Tensor &next_value,
                const MCTSOptions &options={}
            );

            void backup(const MCTSOptions &options={});

            void rootify(float noise_epsilon, const torch::Tensor &noise);

        private:
            int64_t dim;
            float value;
            bool N_is_zero{true};
            torch::Tensor state_, P, Q, N, mask_;
            torch::TensorAccessor<float, 1> Q_accessor;
            torch::TensorAccessor<int64_t, 1> N_accessor;
            std::vector<std::shared_ptr<MCTSNode>> children;

            MCTSNode *parent{nullptr};
            int64_t action{-1};
            bool terminal{false};
            float reward{0.0f};

        private:
            void backup(int64_t action, float value, const MCTSOptions &options);
    };

    void mcts(
        std::vector<std::shared_ptr<MCTSNode>> *root_nodes,
        std::shared_ptr<modules::Base> module,
        std::shared_ptr<rl::simulators::Base> simulator,
        const MCTSOptions &options={}
    );

    std::vector<std::shared_ptr<MCTSNode>> mcts(
        const torch::Tensor &states,
        const std::shared_ptr<rl::policies::constraints::CategoricalMask> masks,
        std::shared_ptr<modules::Base> module,
        std::shared_ptr<rl::simulators::Base> simulator,
        const MCTSOptions &options={}
    );

    std::vector<std::shared_ptr<MCTSNode>> mcts(
        const torch::Tensor &states,
        const torch::Tensor &masks,
        std::shared_ptr<modules::Base> module,
        std::shared_ptr<rl::simulators::Base> simulator,
        const MCTSOptions &options={}
    );
}

#endif /* RL_AGENTS_ALPHA_ZERO_MCTS_H_ */
