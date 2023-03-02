#include "rl/agents/alpha_zero/fast_mcts.h"

#include <rl/cpputils/cpputils.h>
#include <rl/torchutils/repeat.h>

#include "trainer_impl/helpers.h"

using namespace torch::indexing;

namespace rl::agents::alpha_zero
{
    FastMCTSExecutor::FastMCTSExecutor(
        const torch::Tensor &states,
        const torch::Tensor &action_masks,
        std::function<FastMCTSInferenceResult(const torch::Tensor &)> inference_fn,
        std::shared_ptr<rl::simulators::Base> simulator,
        const FastMCTSExecutorOptions &options
    ) : 
        options{options},
        inference_fn{inference_fn},
        simulator{simulator},
        dirchlet_noise_generator{
            options.dirchlet_noise_alpha + torch::zeros({states.size(0), action_masks.size(1)})
        }
    {
        batchsize = states.size(0);
        current_i = batchsize;
        capacity = batchsize;
        action_dim = action_masks.size(1);
        action_dim_vec = torch::arange(
            action_dim,
            torch::TensorOptions{}
                .dtype(torch::kLong)
        );

        init_tensors(states, action_masks);
    }

    FastMCTSExecutor::FastMCTSExecutor(
        const torch::Tensor &states,
        const torch::Tensor &action_masks,
        std::shared_ptr<rl::agents::alpha_zero::modules::Base> module,
        std::shared_ptr<rl::simulators::Base> simulator,
        const FastMCTSExecutorOptions &options
    ) : FastMCTSExecutor{
        states,
        action_masks,
        [module] (const torch::Tensor &states) {
            FastMCTSInferenceResult out{};
            auto module_output = module->forward(states);
            out.probabilities = module_output->policy().get_probabilities();
            out.values = module_output->value_estimates();
            return out;
        },
        simulator,
        options
    }
    {}

    void FastMCTSExecutor::init_tensors(
        const torch::Tensor &states,
        const torch::Tensor &masks
    )
    {
        this->states = states.to(options.module_device).clone();
        this->masks = masks.to(options.module_device).clone();
        actions = - torch::ones({batchsize}, torch::TensorOptions{}.dtype(torch::kLong));
        rewards = torch::zeros({batchsize * action_dim}, torch::TensorOptions{}.dtype(torch::kFloat32));
        terminals = torch::zeros({batchsize * action_dim}, torch::TensorOptions{}.dtype(torch::kBool));
        children = - torch::ones({batchsize * action_dim}, torch::TensorOptions{}.dtype(torch::kLong));
        parents = - torch::ones({batchsize}, torch::TensorOptions{}.dtype(torch::kLong));
        root_indices = torch::arange(batchsize, torch::TensorOptions{}.dtype(torch::kLong));

        auto infer_result = infer(this->states, this->masks);
        P = ((1 - options.dirchlet_noise_epsilon) * infer_result.probabilities.to(torch::kCPU) + options.dirchlet_noise_epsilon * dirchlet_noise_generator.sample()).view({-1});
        Q = torch::zeros({batchsize * action_dim}, torch::TensorOptions{}.dtype(torch::kFloat32));
        N = torch::zeros({batchsize * action_dim}, torch::TensorOptions{}.dtype(torch::kLong));
        V = infer_result.values.clone().to(torch::kCPU);
    }

    void FastMCTSExecutor::expand_node_capacity()
    {
        auto state_expand_size = root_indices.size(0) * options.steps - (capacity - current_i);
        if (state_expand_size <= 0) {
            return;
        }

        auto action_expand_size = action_dim * state_expand_size;
        capacity += state_expand_size;

        std::vector<int64_t> state_repeat_vector{};
        state_repeat_vector.push_back(state_expand_size);
        for (int i = 1; i < states.sizes().size(); i++) {
            state_repeat_vector.push_back(1);
        }

        states = torch::concat({
            states,
            rl::torchutils::repeat(torch::zeros_like(states.index({0})).unsqueeze_(0), {state_expand_size})
        });
        masks = torch::concat({
            masks,
            torch::zeros_like(masks.index({0})).unsqueeze_(0).repeat({state_expand_size, 1})
        });
        actions = torch::concat({
            actions,
            - torch::ones({state_expand_size}, actions.options())
        });
        V = torch::concat({
            N,
            torch::zeros({state_expand_size}, V.options())
        });
        parents = torch::concat({
            parents,
            - torch::ones({state_expand_size}, parents.options())
        });
        rewards = torch::concat({
            rewards,
            - torch::ones({action_expand_size}, rewards.options())
        });
        terminals = torch::concat({
            terminals,
            torch::zeros({action_expand_size}, terminals.options())
        });
        children = torch::concat({
            children,
            - torch::ones({action_expand_size}, children.options())
        });
        P = torch::concat({
            P,
            torch::zeros({action_expand_size}, P.options())
        });
        Q = torch::concat({
            Q,
            torch::zeros({action_expand_size}, Q.options())
        });
        N = torch::concat({
            N,
            torch::zeros({action_expand_size}, N.options())
        });
    }

    FastMCTSInferenceResult FastMCTSExecutor::infer(
        const torch::Tensor &states, const torch::Tensor &masks
    )
    {
        auto out = inference_fn(states);
        out.probabilities = torch::where(
            masks,
            out.probabilities,
            torch::zeros_like(out.probabilities)
        );
        out.probabilities /= out.probabilities.sum(1, true);

        return out;
    }

    void FastMCTSExecutor::step(const torch::Tensor &actions)
    {
        if (actions.size(0) != root_indices.size(0)) {
            throw std::invalid_argument{"Expecting one action per non-terminal episode."};
        }
        if (root_indices.size(0) == 0) {
            throw std::runtime_error{"All episodes are terminal."};
        }

        auto action_indices = node_actions_to_action_indices(root_indices, actions);
        root_indices = children.index({action_indices});
        auto terminals = this->terminals.index({action_indices});
        root_indices = root_indices.index({~terminals});
        parents.index_put_({root_indices}, -1);
        
        auto next_action_indices = node_indices_to_action_indices(root_indices);
        P.index_put_(
            {next_action_indices},
            (1 - options.dirchlet_noise_epsilon) * P.index({next_action_indices})
            + options.dirchlet_noise_epsilon * dirchlet_noise_generator.sample().index({Slice(None, root_indices.size(0))}).view({-1})
        );

        steps++;
    }

    void FastMCTSExecutor::run()
    {
        expand_node_capacity();
        for (int64_t i = 0; i < options.steps; i++)
        {
            auto select_result = select(root_indices);
            expand(select_result.nodes, select_result.actions);
            backup(select_result.nodes, select_result.actions);
        }
    }

    FastMCTSExecutor::SelectResult FastMCTSExecutor::select(const torch::Tensor &current_nodes)
    {
        auto all_action_indices = node_indices_to_action_indices(current_nodes);

        auto n = N.index({all_action_indices}).view({current_nodes.size(0), action_dim});
        auto nsum = n.sum(1);
        auto p = P.index({all_action_indices}).view({current_nodes.size(0), action_dim});
        auto q = Q.index({all_action_indices}).view({current_nodes.size(0), action_dim});

        auto puct = q + p * nsum.sqrt() / (1 + n) * (options.c1 * ((nsum + options.c2 + 1.0f) / options.c2).log_());
        auto actions = torch::where(
            nsum == 0,
            p.argmax(1),
            puct.argmax(1)
        );
        auto action_indices = node_actions_to_action_indices(current_nodes, actions);

        auto next_nodes = children.index({action_indices});
        auto not_end_of_road = (next_nodes != -1).logical_and(~this->terminals.index({action_indices}));

        SelectResult out{};
        out.nodes = current_nodes;
        out.actions = actions;

        if (not_end_of_road.any().item().toBool()) {
            auto continued_road = select( next_nodes.index({not_end_of_road}) );
            out.nodes = out.nodes.index_put({not_end_of_road}, continued_road.nodes);
            out.actions = out.actions.index_put({not_end_of_road}, continued_road.actions);
        }

        return out;
    }

    void FastMCTSExecutor::expand(const torch::Tensor &nodes_, const torch::Tensor &actions_)
    {
        auto action_indices = node_actions_to_action_indices(nodes_, actions_);
        auto terminals = this->terminals.index({action_indices});
        auto inv_terminals = ~terminals;

        if (terminals.all().item().toBool()) {
            return;
        }

        auto nodes = nodes_.index({inv_terminals});
        auto actions = actions_.index({inv_terminals});
        action_indices = action_indices.index({inv_terminals});
        auto states = this->states.index({nodes});

        auto observations = simulator->step(states, actions);
        auto next_node_indices = current_i + torch::arange(nodes.size(0), nodes.options());
        auto next_action_indices = node_indices_to_action_indices(next_node_indices);
        current_i += nodes.size(0);

        auto next_states = observations.next_states.states;
        auto next_masks = trainer_impl::get_mask(*observations.next_states.action_constraints);

        auto inference_result = infer(next_states, next_masks);

        this->states.index_put_({next_node_indices}, next_states);
        this->masks.index_put_({next_node_indices}, next_masks);
        this->actions.index_put_({next_node_indices}, actions);
        this->rewards.index_put_({action_indices}, observations.rewards);
        this->terminals.index_put_({action_indices}, observations.terminals);
        this->children.index_put_({action_indices}, next_node_indices);
        this->parents.index_put_({next_node_indices}, nodes);
        this->P.index_put_({next_action_indices}, inference_result.probabilities.view({-1}));
        this->V.index_put_({next_node_indices}, inference_result.values);
    }

    void FastMCTSExecutor::backup(const torch::Tensor &nodes_, const torch::Tensor &actions_)
    {
        auto nodes = nodes_;
        auto actions = actions_;
        auto action_indices = node_actions_to_action_indices(nodes, actions);
        auto values = (~this->terminals.index({action_indices})) * this->V.index({this->children.index({action_indices})});

        auto is_node = nodes != -1;

        while (is_node.any().item().toBool())
        {
            nodes = nodes.index({is_node});
            actions = actions.index({is_node});
            action_indices = node_actions_to_action_indices(nodes, actions);
            values = this->rewards.index({action_indices}) + options.discount * values.index({is_node});

            auto n = N.index({action_indices});

            Q.index_put_(
                {action_indices},
                (n * Q.index({action_indices}) + values) / (n + 1)
            );
            N.index_put_(
                {action_indices},
                n + 1
            );

            // Set actions of next iteration to whatever action led to the current node.
            actions = this->actions.index({nodes});
            // Set parents.
            nodes = this->parents.index({nodes});
            is_node = nodes != -1;
        }
    }

    FastMCTSEpisodes FastMCTSExecutor::get_episodes()
    {
        if (!all_terminals()) {
            throw std::runtime_error{"Cannot get episodes until all episodes are terminal."};
        }

        FastMCTSEpisodes out{};

        out.states = rl::torchutils::repeat(torch::zeros_like(states.index({0})).unsqueeze(0).unsqueeze(1), {batchsize, steps});
        out.masks = torch::zeros({batchsize, steps, action_dim}, torch::TensorOptions{}.dtype(torch::kBool).device(options.sim_device));
        out.actions = torch::zeros({batchsize, steps}, torch::TensorOptions{}.dtype(torch::kLong).device(options.sim_device));
        out.rewards = torch::zeros({batchsize, steps}, torch::TensorOptions{}.dtype(torch::kFloat32).device(options.sim_device));
        out.lengths = torch::zeros({batchsize}, torch::TensorOptions{}.dtype(torch::kLong).device(options.sim_device));

        
    }
}
