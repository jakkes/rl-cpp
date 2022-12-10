#include "rl/agents/alpha_zero/mcts.h"


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

    MCTSSelectResult MCTSNode::select(const MCTSOptions &options)
    {
        if (terminal) {
            MCTSSelectResult out{};
            out.node = parent;
            out.action = this->action;
            return out;
        }

        auto puct = Q + P * N.sum().sqrt() / (1 + N) * (options.c1 + ((N.sum() + options.c2 + 1.0f) / options.c2).log_());
        puct = torch::where(mask_, puct, torch::zeros_like(puct) - INFINITY);
        auto action = torch::argmax(puct).item().toLong();
        
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
            assert(children[action]->terminal);
            assert(children[action]->reward == reward);
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
        next_node->reward = reward;
        next_node->terminal = terminal;
    }

    void MCTSNode::backup(const MCTSOptions &options)
    {
        if (!parent) {
            throw std::runtime_error{"Cannot backup from root node."};
        }

        if (terminal) {
            parent->backup(action, reward, options);
        }
        else {
            parent->backup(action, reward + options.discount * value, options);
        }
    }

    void MCTSNode::backup(int64_t action, float value, const MCTSOptions &options)
    {
        Q_accessor[action] = (N_accessor[action] * Q_accessor[action] + value) / (N_accessor[action] + 1);
        N_accessor[action] = N_accessor[action] + 1;

        if (!parent) {
            return;
        }

        parent->backup(this->action, reward + options.discount * value, options);
    }

    void mcts(
        std::vector<std::shared_ptr<MCTSNode>> *root_nodes_,
        std::shared_ptr<modules::Base> module,
        std::shared_ptr<rl::simulators::Base> simulator,
        const MCTSOptions &options
    )
    {
        torch::InferenceMode inference_guard{};

        auto &root_nodes{*root_nodes_};
        for (auto &node : root_nodes) {
            node->rootify();
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

            auto module_output = module->forward(observation.next_states.states);
            auto policy = module_output->policy();
            auto value = module_output->value_estimates();
            policy.include(observation.next_states.action_constraints);

            for (int i = 0; i < root_nodes.size(); i++) {
                select_results[i].node->expand(
                    select_results[i].action,
                    observation.rewards.index({i}).item().toFloat(),
                    observation.terminals.index({i}).item().toBool(),
                    observation.next_states.states.index({i}),
                    next_masks.index({i}),
                    policy.get_probabilities().index({i}),
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
        const torch::Tensor &masks,
        std::shared_ptr<modules::Base> module,
        std::shared_ptr<rl::simulators::Base> simulator,
        const MCTSOptions &options
    )
    {
        return mcts(
            states,
            std::make_shared<rl::policies::constraints::CategoricalMask>(masks),
            module,
            simulator,
            options
        );
    }

    std::vector<std::shared_ptr<MCTSNode>> mcts(
        const torch::Tensor &states,
        std::shared_ptr<rl::policies::constraints::CategoricalMask> masks,
        std::shared_ptr<modules::Base> module,
        std::shared_ptr<rl::simulators::Base> simulator,
        const MCTSOptions &options
    )
    {
        torch::InferenceMode inference_guard{};

        auto output = module->forward(states.to(options.module_device));
        auto prior_policy = output->policy();
        prior_policy.include(masks);

        auto priors = prior_policy.get_probabilities();
        auto values = output->value_estimates();

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

        mcts(&root_nodes, module, simulator, options);

        return root_nodes;
    }
}
