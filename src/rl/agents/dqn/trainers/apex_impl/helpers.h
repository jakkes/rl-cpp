#ifndef RL_AGENTS_DQN_TRAINERS_APEX_IMPL_HELPERS_H_
#define RL_AGENTS_DQN_TRAINERS_APEX_IMPL_HELPERS_H_


#include <vector>
#include <memory>

#include <torch/torch.h>

#include <rl/buffers/tensor.h>
#include <rl/env/base.h>
#include <rl/policies/constraints/categorical_mask.h>
#include <rl/agents/dqn/trainers/apex.h>

namespace rl::agents::dqn::trainers::apex_impl
{
    static
    std::shared_ptr<rl::buffers::Tensor> create_buffer(
        int64_t capacity,
        std::shared_ptr<rl::env::Factory> env_factory,
        const rl::agents::dqn::trainers::ApexOptions &options
    )
    {
        auto env = env_factory->get();
        env->set_logger(nullptr);

        auto state = env->reset();
        auto mask_constraint = dynamic_cast<const rl::policies::constraints::CategoricalMask&>(*state->action_constraint);

        std::vector<std::vector<int64_t>> tensor_shapes{};
        tensor_shapes.push_back(state->state.sizes().vec());   // States
        tensor_shapes.push_back(mask_constraint.mask().sizes().vec()); // Masks
        tensor_shapes.push_back({});   // Actions
        tensor_shapes.push_back({});   // Rewards
        tensor_shapes.push_back({});   // Not terminals
        tensor_shapes.push_back(state->state.sizes().vec());   // Next states
        tensor_shapes.push_back(mask_constraint.mask().sizes().vec()); // Next masks

        std::vector<torch::TensorOptions> tensor_options{};
        tensor_options.push_back(state->state.options().device(options.replay_device));
        tensor_options.push_back(mask_constraint.mask().options().device(options.replay_device));
        tensor_options.push_back(torch::TensorOptions{}.dtype(torch::kLong).device(options.replay_device));
        tensor_options.push_back(torch::TensorOptions{}.dtype(torch::kFloat32).device(options.replay_device));
        tensor_options.push_back(torch::TensorOptions{}.dtype(torch::kBool).device(options.replay_device));
        tensor_options.push_back(state->state.options().device(options.replay_device));
        tensor_options.push_back(mask_constraint.mask().options().device(options.replay_device));

        auto buffer = std::make_shared<rl::buffers::Tensor>(
            capacity,
            tensor_shapes,
            tensor_options
        );

        return buffer;
    }

    inline
    torch::Tensor get_mask(const rl::policies::constraints::Base &constraint) {
        return dynamic_cast<const rl::policies::constraints::CategoricalMask&>(constraint).mask();
    }
}

#endif /* RL_AGENTS_DQN_TRAINERS_APEX_IMPL_HELPERS_H_ */
