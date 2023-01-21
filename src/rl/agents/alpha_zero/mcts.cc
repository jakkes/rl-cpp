#include "rl/agents/alpha_zero/mcts.h"

#include <rl/policies/dirchlet.h>


namespace rl::agents::alpha_zero
{
    static auto N_options = torch::TensorOptions{}.dtype(torch::kLong);

    MCTSNode::MCTSNode(
        const torch::Tensor &state,
        const torch::Tensor &mask,
        const torch::Tensor &prior,
        float value
    ) 
        :
        state_{state},
        mask_{mask},
        value{value},
        P{prior.to(torch::kCPU)},
        Q{torch::zeros_like(P)},
        Q_accessor{Q.accessor<float, 1>()},
        N{torch::zeros({Q.size(0)}, N_options)},
        N_accessor{N.accessor<int64_t, 1>()}
    {
        dim = prior.size(0);
        children.resize(dim);
    }

    void MCTSNode::rootify(float noise_epsilon, const torch::Tensor &noise) {
        action = -1;
        parent = nullptr;

        P = (1 - noise_epsilon) * P + noise_epsilon * noise;
    }

    MCTSSelectResult MCTSNode::select(const MCTSOptions &options)
    {
        if (terminal_) {
            MCTSSelectResult out{};
            out.node = parent;
            out.action = this->action;
            return out;
        }

        int64_t action;

        if (N_is_zero) {
            action = P.argmax().item().toLong();
        }
        else {
            auto puct = Q + P * N.sum().sqrt() / (1 + N) * (options.c1 + ((N.sum() + options.c2 + 1.0f) / options.c2).log_());
            puct = torch::where(mask_, puct, torch::zeros_like(puct) - INFINITY);
            action = torch::argmax(puct).item().toLong();
        }
        
        if (children[action]) {
            return children[action]->select(options);
        }

        MCTSSelectResult out{};
        out.node = this;
        out.action = action;
        return out;
    }

    void MCTSNode::expand(
        int64_t action,
        float reward,
        bool terminal,
        const torch::Tensor &next_state,
        const torch::Tensor &next_mask,
        const torch::Tensor &next_prior,
        const torch::Tensor &next_value,
        const MCTSOptions &options
    )
    {
        if (children[action]) {
            // Action was already expanded. This may happen if action resulted in a
            // terminal state.
            assert(children[action]->terminal_);
            assert(children[action]->reward_ == reward);
            assert(terminal);
            return;
        }

        auto next_node = std::make_shared<MCTSNode>(
            next_state,
            next_mask,
            next_prior,
            next_value.item().toFloat()
        );
        children[action] = next_node;
        next_node->parent = this;
        next_node->action = action;
        next_node->reward_ = reward;
        next_node->terminal_ = terminal;
    }

    void MCTSNode::backup(const MCTSOptions &options)
    {
        if (!parent) {
            throw std::runtime_error{"Cannot backup from root node."};
        }

        if (terminal_) {
            parent->backup(action, reward_, options);
        }
        else {
            parent->backup(action, reward_ + options.discount * value, options);
        }
    }

    void MCTSNode::backup(int64_t action, float value, const MCTSOptions &options)
    {
        Q_accessor[action] = (N_accessor[action] * Q_accessor[action] + value) / (N_accessor[action] + 1);
        N_accessor[action] = N_accessor[action] + 1;

        if (N_is_zero) {
            N_is_zero = false;
        }

        if (!parent) {
            return;
        }

        parent->backup(this->action, reward_ + options.discount * value, options);
    }

    void mcts(
        std::vector<std::shared_ptr<MCTSNode>> *root_nodes_,
        std::function<MCTSInferenceResult(const torch::Tensor &)> inference_fn,
        std::shared_ptr<rl::simulators::Base> simulator,
        const MCTSOptions &options
    )
    {
        auto &root_nodes{*root_nodes_};

        int64_t batchsize = root_nodes.size();
        int64_t dim = root_nodes.front()->mask().size(0);

        rl::policies::Dirchlet dirchlet_distribution{
            options.dirchlet_noise_alpha + torch::zeros({batchsize, dim})
        };
        auto dirchlet_noise = dirchlet_distribution.sample();

        for (int i = 0; i < batchsize; i++) {
            root_nodes[i]->rootify(options.dirchlet_noise_epsilon, dirchlet_noise.index({i}));
        }

        std::vector<MCTSSelectResult> select_results{};
        select_results.resize(root_nodes.size());


        for (int step = 0; step < options.steps; step++) {
            for (int i = 0; i < root_nodes.size(); i++) {
                select_results[i] = root_nodes[i]->select(options);
            }

            std::vector<torch::Tensor> states{}; states.reserve(root_nodes.size());
            std::vector<int64_t> actions{}; actions.reserve(root_nodes.size());
            for (int i = 0; i < root_nodes.size(); i++) {
                states.push_back(select_results[i].node->state());
                actions.push_back(select_results[i].action);
            }

            auto observation = simulator->step(
                torch::stack(states, 0),
                torch::tensor(actions, torch::TensorOptions{}.dtype(torch::kLong))
            );

            auto next_masks = std::dynamic_pointer_cast<rl::policies::constraints::CategoricalMask>(observation.next_states.action_constraints)->mask();

            auto output = inference_fn(observation.next_states.states.to(options.module_device));
            auto priors = output.policies.get_probabilities().to(torch::kCPU);
            auto value = output.values.to(torch::kCPU);

            for (int i = 0; i < root_nodes.size(); i++) {
                select_results[i].node->expand(
                    select_results[i].action,
                    observation.rewards.index({i}).item().toFloat(),
                    observation.terminals.index({i}).item().toBool(),
                    observation.next_states.states.index({i}),
                    next_masks.index({i}),
                    priors.index({i}),
                    value.index({i}),
                    options
                );
            }

            for (int i = 0; i < root_nodes.size(); i++) {
                select_results[i].node->get_child(select_results[i].action)->backup();
            }
        }
    }

    std::vector<std::shared_ptr<MCTSNode>> mcts(
        const torch::Tensor &states,
        std::shared_ptr<rl::policies::constraints::CategoricalMask> masks,
        std::function<MCTSInferenceResult(const torch::Tensor &)> inference_fn,
        std::shared_ptr<rl::simulators::Base> simulator,
        const MCTSOptions &options
    )
    {
        auto output = inference_fn(states.to(options.module_device));
        output.policies.include(masks);

        auto priors = output.policies.get_probabilities().to(torch::kCPU);
        auto values = output.values.to(torch::kCPU);

        std::vector<std::shared_ptr<MCTSNode>> root_nodes{};
        root_nodes.resize(priors.size(0));

        for (int i = 0; i < root_nodes.size(); i++) {
            root_nodes[i] = std::make_shared<MCTSNode>(
                states.index({i}),
                masks->mask().index({i}),
                priors.index({i}),
                values.index({i}).item().toFloat()
            );
        }

        mcts(&root_nodes, inference_fn, simulator, options);

        return root_nodes;
    }
}
