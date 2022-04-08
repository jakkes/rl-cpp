target_sources(
    rl
    PRIVATE
        buffers/state_transition.cc
        buffers/tensor.cc

        cpputils/metronome.cc

        env/cart_pole.cc
        env/base.cc

        logging/client/base.cc
        logging/client/ema.cc
        
        policies/categorical.cc
        policies/beta.cc
        policies/normal.cc
        policies/gamma.cc
        policies/factorized.cc

            policies/constraints/stack.cc
            policies/constraints/categorical_mask.cc
            policies/constraints/box.cc

        torchutils.cc
)

add_subdirectory(agents)
