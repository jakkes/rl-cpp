#ifndef RL_AGENTS_ALPHA_ZERO_FAST_MCTS_H_
#define RL_AGENTS_ALPHA_ZERO_FAST_MCTS_H_

#include <memory>
#include <functional>

#include <torch/torch.h>
#include <rl/simulators/base.h>
#include <rl/agents/alpha_zero/modules/base.h>
#include <rl/option.h>
#include <rl/policies/dirchlet.h>


namespace rl::agents::alpha_zero
{
    struct FastMCTSExecutorOptions
    {
        RL_OPTION(torch::Device, module_device) = torch::kCPU;
        RL_OPTION(torch::Device, sim_device) = torch::kCPU;

        RL_OPTION(float, c1) = 1.25f;
        RL_OPTION(float, c2) = 19652;

        RL_OPTION(float, discount) = 1.0f;
        RL_OPTION(int, steps) = 100;

        RL_OPTION(float, dirchlet_noise_alpha) = 0.1f;
        RL_OPTION(float, dirchlet_noise_epsilon) = 0.5f;
    };

    struct FastMCTSInferenceResult
    {
        torch::Tensor probabilities;
        torch::Tensor values;
    };

    struct FastMCTSEpisodes
    {
        torch::Tensor states;
        torch::Tensor masks;
        torch::Tensor actions;
        torch::Tensor rewards;
        torch::Tensor lengths;
    };

    class FastMCTSExecutor
    {
        public:
            FastMCTSExecutor(
                const torch::Tensor &states,
                const torch::Tensor &action_masks,
                std::function<FastMCTSInferenceResult(const torch::Tensor &)> inference_fn,
                std::shared_ptr<rl::simulators::Base> simulator,
                const FastMCTSExecutorOptions &options={}
            );

            FastMCTSExecutor(
                const torch::Tensor &states,
                const torch::Tensor &action_masks,
                std::shared_ptr<rl::agents::alpha_zero::modules::Base> module,
                std::shared_ptr<rl::simulators::Base> simulator,
                const FastMCTSExecutorOptions &options={}
            );

            inline
            const torch::Tensor current_visit_counts() const {
                return N.index({node_indices_to_action_indices(root_indices)}).view({root_indices.size(0), action_dim});
            }

            inline
            bool all_terminals() const {
                return root_indices.size(0) == 0;
            }

            void step(const torch::Tensor &actions);

            void run();

            FastMCTSEpisodes get_episodes();

        private:
            struct SelectResult
            {
                torch::Tensor nodes;
                torch::Tensor actions;
            };

        private:
            const FastMCTSExecutorOptions options;
            std::function<FastMCTSInferenceResult(const torch::Tensor &)> inference_fn;
            std::shared_ptr<rl::simulators::Base> simulator;

            rl::policies::Dirchlet dirchlet_noise_generator;

            torch::Tensor states;
            torch::Tensor masks;
            torch::Tensor actions;
            torch::Tensor rewards;
            torch::Tensor terminals;
            torch::Tensor children;
            torch::Tensor parents;
            torch::Tensor root_indices;
            torch::Tensor P;
            torch::Tensor Q;
            torch::Tensor N;
            torch::Tensor V;

            int64_t batchsize;
            int64_t action_dim;
            torch::Tensor action_dim_vec;
            int64_t current_i;
            int64_t capacity;
            int64_t steps{0};
            torch::Tensor step_actions;

        private:
            void init_tensors(
                const torch::Tensor &states,
                const torch::Tensor &action_masks
            );
            void expand_node_capacity();

            FastMCTSInferenceResult infer(const torch::Tensor &states, const torch::Tensor &masks);

            inline
            torch::Tensor node_indices_to_action_indices(const torch::Tensor &node_indices) const {
                return (action_dim * node_indices.view({-1, 1}) + action_dim_vec.view({1, -1})).view({-1});
            }

            inline
            torch::Tensor node_actions_to_action_indices(const torch::Tensor &node_indices, const torch::Tensor &actions) const {
                return action_dim * node_indices + actions;
            }

            SelectResult select(const torch::Tensor &current_nodes);
            void expand(const torch::Tensor &nodes, const torch::Tensor &actions);
            void backup(const torch::Tensor &nodes, const torch::Tensor &actions);
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_FAST_MCTS_H_ */
