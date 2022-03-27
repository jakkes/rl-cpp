target_sources(
    rl
    PRIVATE
        agents/ppo/agent.cc
            agents/ppo/trainers/basic.cc

        buffers/state_transition.cc
        buffers/tensor.cc

        env/cart_pole.cc
        env/base.cc

        logging/client/base.cc
        logging/client/ema.cc
        
        policies/categorical.cc
        policies/beta.cc

            policies/constraints/stack.cc
            policies/constraints/categorical_mask.cc
            policies/constraints/box.cc

        torchutils.cc
)
