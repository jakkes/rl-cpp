target_sources(
    rl
    PUBLIC
        agents/ppo/agent.cc
            agents/ppo/trainers/basic.cc

        env/cart_pole.cc
        
        policies/categorical.cc
        policies/rejection_sampler.cc
            policies/constraints/stack.cc
            policies/constraints/cat.cc
            policies/constraints/categorical_mask.cc

        buffers/state_transition.cc
        buffers/tensor.cc

        torchutils.cc
)
