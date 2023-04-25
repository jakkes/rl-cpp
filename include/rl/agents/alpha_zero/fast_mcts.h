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

    /**
     * @brief Network inference result.
     */
    struct FastMCTSInferenceResult
    {
        // Policy (prior), of shape (N, M), where N is batch size and M action dimension.
        torch::Tensor probabilities;
        // Values, of shape (N, )
        torch::Tensor values;
    };

    /**
     * @brief A batch of episodes.
     */
    struct FastMCTSEpisodes
    {
        // States, of shape (N, S, *), where N is batch size, and S is the max episode length.
        torch::Tensor states;
        // Action masks, of shape (N, S, M), where M is action dim.
        torch::Tensor masks;
        // Actions, of shape (N, S)
        torch::Tensor actions;
        // Rewards, of shape (N, S)
        torch::Tensor rewards;
        // Episode lengths, of shape (N, )
        torch::Tensor lengths;
    };

    /**
     * @brief Batch MCTS executor.
     * 
     * This MCTS runner will execute a batch of episodes in parallell. However, batch
     * size will shrink towards the end of episodes incase not all episodes have equal
     * length.
     * 
     */
    class FastMCTSExecutor
    {
        public:

            /**
             * @brief Construct a new FastMCTSExecutor object
             * 
             * @param states Start states
             * @param action_masks Start action masks
             * @param inference_fn Function accepting a state, producing an inference result
             * @param simulator Simulator
             * @param options MCTS options
             */
            FastMCTSExecutor(
                const torch::Tensor &states,
                const torch::Tensor &action_masks,
                std::function<FastMCTSInferenceResult(const torch::Tensor &)> inference_fn,
                std::shared_ptr<rl::simulators::Base> simulator,
                const FastMCTSExecutorOptions &options={}
            );

            /**
             * @brief Construct a new FastMCTSExecutor object
             * 
             * @param states Start states
             * @param action_masks Start action masks
             * @param module Alpha zero network
             * @param simulator Simulator
             * @param options MCTS options
             */
            FastMCTSExecutor(
                const torch::Tensor &states,
                const torch::Tensor &action_masks,
                std::shared_ptr<rl::agents::alpha_zero::modules::Base> module,
                std::shared_ptr<rl::simulators::Base> simulator,
                const FastMCTSExecutorOptions &options={}
            );

            /**
             * @brief Returns the visit counts of the current nodes.
             * 
             * @return const torch::Tensor Tensor of shape (N, M), where N is the number
             * of still running episodes, and M the action dim size.
             */
            inline const torch::Tensor current_visit_counts() const {
                return N.index({node_indices_to_action_indices(root_indices)}).view({root_indices.size(0), action_dim});
            }

            /**
             * @return true if all episodes have terminated
             * @return false if any episode is still not terminal
             */
            inline bool all_terminals() const {
                return root_indices.size(0) == 0;
            }

            /**
             * @brief Perform an actual step in the MCTS search tree. This will move
             * the current node forward, possibly terminating a set of episodes.
             * 
             * @param actions Actions, shape (N, ), where N is the number of non-terminal
             * episodes.
             */
            void step(const torch::Tensor &actions);

            /**
             * @brief Executes a forward pass of the MCTS search algorithm. This will
             * run make N hypothetical steps in the search tree, where N is defined in
             * `options.steps`.
             * 
             */
            void run();

            /**
             * @return FastMCTSEpisodes all episodes. Only callable once all episodes
             * have terminated.
             */
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

            // State tensors
            torch::Tensor states;
            // Action mask tensors
            torch::Tensor masks;
            // Reward obtained from taking an action from the state
            torch::Tensor rewards;
            // If an action from the state is terminal
            torch::Tensor terminals;
            // Index of next state when following an action
            torch::Tensor children;
            // Index of parent state
            torch::Tensor parents;
            // Action from parent that resulted in state
            torch::Tensor step_actions;

            // Current node indices
            torch::Tensor root_indices;
            
            // Prior in a state
            torch::Tensor P;
            // Accumulated value following an action in a state
            torch::Tensor Q;
            // Visit count to an action in a state
            torch::Tensor N;
            // Value of a state
            torch::Tensor V;

            // Number of episodes run in batch
            int64_t batchsize;
            // Action dim
            int64_t action_dim;
            // Vector of [0, 1, ..., action_dim-1]
            torch::Tensor action_dim_vec;
            // Cursor to next state insertion point
            int64_t current_i;
            // Current state capacity
            int64_t capacity;
            // Number of steps executed. This is number of calls to `step()`, number of
            // mcts steps is this value times options.steps.
            int64_t steps{0};

        private:

            // Init tensors, called from constructor
            void init_tensors(
                const torch::Tensor &states,
                const torch::Tensor &action_masks
            );

            // Expands the capacity to cover any search path.
            void expand_node_capacity();

            // Run inference
            FastMCTSInferenceResult infer(const torch::Tensor &states, const torch::Tensor &masks);

            // Get all action indices for the given node indices
            inline torch::Tensor node_indices_to_action_indices(const torch::Tensor &node_indices) const {
                return (action_dim * node_indices.view({-1, 1}) + action_dim_vec.view({1, -1})).view({-1});
            }

            // Get action indices of actions for given nodes.
            inline torch::Tensor node_actions_to_action_indices(const torch::Tensor &node_indices, const torch::Tensor &actions) const {
                return action_dim * node_indices + actions;
            }

            // Run MCTS select step
            SelectResult select(const torch::Tensor &current_nodes);
            // Run MCTS expand step
            void expand(const torch::Tensor &nodes, const torch::Tensor &actions);
            // Run MCTS backup step
            void backup(const torch::Tensor &nodes, const torch::Tensor &actions);
    };
}

#endif /* RL_AGENTS_ALPHA_ZERO_FAST_MCTS_H_ */
