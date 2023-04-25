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
        this->states = states.to(options.sim_device).clone();
        this->masks = masks.to(options.sim_device).clone();
        rewards = torch::zeros({batchsize * action_dim}, torch::TensorOptions{}.dtype(torch::kFloat32));
        terminals = torch::zeros({batchsize * action_dim}, torch::TensorOptions{}.dtype(torch::kBool));
        children = - torch::ones({batchsize * action_dim}, torch::TensorOptions{}.dtype(torch::kLong));
        parents = - torch::ones({batchsize}, torch::TensorOptions{}.dtype(torch::kLong));
        step_actions = - torch::ones({0}, torch::TensorOptions{}.dtype(torch::kLong));

        root_indices = torch::arange(batchsize, torch::TensorOptions{}.dtype(torch::kLong));

        auto infer_result = infer(this->states.to(options.module_device), this->masks.to(options.module_device));
        P = ((1 - options.dirchlet_noise_epsilon) * infer_result.probabilities.to(torch::kCPU) + options.dirchlet_noise_epsilon * dirchlet_noise_generator.sample()).view({-1});
        Q = torch::zeros({batchsize * action_dim}, torch::TensorOptions{}.dtype(torch::kFloat32));
        N = torch::zeros({batchsize * action_dim}, torch::TensorOptions{}.dtype(torch::kLong));
        V = infer_result.values.to(torch::kCPU);
        
    }

    void FastMCTSExecutor::expand_node_capacity()
    {
        auto state_expand_size = root_indices.size(0) * options.steps - (capacity - current_i);
        if (state_expand_size <= 0) {
            return;
        }

        auto action_expand_size = action_dim * state_expand_size;
        capacity += state_expand_size;

        states = torch::concat({
            states,
            rl::torchutils::repeat(torch::zeros_like(states.index({0})).unsqueeze(0), {state_expand_size})
        });
        masks = torch::concat({
            masks,
            torch::zeros_like(masks.index({0})).unsqueeze(0).repeat({state_expand_size, 1})
        });
        V = torch::concat({
            V,
            torch::zeros({state_expand_size}, V.options())
        });
        parents = torch::concat({
            parents,
            - torch::ones({state_expand_size}, parents.options())
        });
        rewards = torch::concat({
            rewards,
            torch::zeros({action_expand_size}, rewards.options())
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
        torch::NoGradGuard no_grad_guard{};
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

        step_actions = torch::concat({step_actions, actions}, 0);

        auto action_indices = node_actions_to_action_indices(root_indices, actions);
        root_indices = children.index({action_indices});
        auto terminals = this->terminals.index({action_indices});
        root_indices = root_indices.index({~terminals});
        
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

        auto puct = q + p * nsum.unsqueeze(1).sqrt() / (1 + n) * (options.c1 * ((nsum.unsqueeze(1) + options.c2 + 1.0f) / options.c2).log_());
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
        auto not_terminals = ~terminals;

        if (terminals.all().item().toBool()) {
            return;
        }

        auto nodes = nodes_.index({not_terminals});
        auto actions = actions_.index({not_terminals});
        action_indices = action_indices.index({not_terminals});
        auto states = this->states.index({nodes});

        auto observations = simulator->step(states, actions);
        auto next_node_indices = current_i + torch::arange(nodes.size(0), nodes.options());
        auto next_action_indices = node_indices_to_action_indices(next_node_indices);
        current_i += nodes.size(0);

        auto next_states = observations.next_states.states;
        auto next_masks = trainer_impl::get_mask(*observations.next_states.action_constraints);

        auto inference_result = infer(next_states.to(options.module_device), next_masks.to(options.module_device));

        this->states.index_put_({next_node_indices}, next_states);
        this->masks.index_put_({next_node_indices}, next_masks);
        this->rewards.index_put_({action_indices}, observations.rewards);
        this->terminals.index_put_({action_indices}, observations.terminals);
        this->children.index_put_({action_indices}, next_node_indices);
        this->parents.index_put_({next_node_indices}, nodes);
        this->P.index_put_({next_action_indices}, inference_result.probabilities.view({-1}).to(torch::kCPU));
        this->V.index_put_({next_node_indices}, inference_result.values.to(torch::kCPU));
    }

    void FastMCTSExecutor::backup(const torch::Tensor &nodes_, const torch::Tensor &actions_)
    {
        auto nodes = nodes_;
        auto actions = actions_;
        auto child_indices = this->children.index({node_actions_to_action_indices(nodes, actions)});
        auto values = this->V.index({child_indices});
        auto root_indices = this->root_indices;
        

        while (true)
        {
            auto action_indices = node_actions_to_action_indices(nodes, actions);
            values = this->rewards.index({action_indices}) + options.discount * (~this->terminals.index({action_indices})) * values;

            auto n = N.index({action_indices});

            Q.index_put_(
                {action_indices},
                (n * Q.index({action_indices}) + values) / (n + 1)
            );
            N.index_put_(
                {action_indices},
                n + 1
            );

            // Prepare for next round of backup
            auto nodes_to_backup = nodes != root_indices;
            if (!nodes_to_backup.any().item().toBool()) {
                break;
            }

            // Set actions of next iteration to whatever action led to the current node.
            actions = this->actions.index({nodes}).index({nodes_to_backup});
            nodes = this->parents.index({nodes}).index({nodes_to_backup});
            root_indices = root_indices.index({nodes_to_backup});
            values = values.index({nodes_to_backup});
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

        auto nodes = torch::arange(batchsize);
        auto batchvec = nodes;
        auto step_vec = torch::zeros({batchsize}, torch::TensorOptions{}.dtype(torch::kLong));
        int64_t step_action_offset{0};

        for (int i = 0; i < steps; i++) {
            out.states.index_put_({batchvec, step_vec}, this->states.index({nodes}));
            out.masks.index_put_({batchvec, step_vec}, this->masks.index({nodes}));

            auto actions = this->step_actions.index({step_action_offset + torch::arange(nodes.size(0))});
            auto action_indices = node_actions_to_action_indices(nodes, actions);
            out.actions.index_put_({batchvec, step_vec}, actions);
            out.rewards.index_put_({batchvec, step_vec}, this->rewards.index({action_indices}));

            auto terminals = this->terminals.index({action_indices});
            auto non_terminals = ~terminals;
            if (terminals.any().item().toBool()) {
                out.lengths.index_put_({batchvec.index({terminals})}, i + 1);
            }

            nodes = this->children.index({action_indices.index({non_terminals})});
            batchvec = batchvec.index({non_terminals});
            step_vec = step_vec.index({non_terminals}) + 1;
            step_action_offset += nodes.size(0);

            assert (!(nodes == -1).any().item().toBool());
        }

        return out;
    }
}
