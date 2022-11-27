target_sources(
    rl
    PRIVATE
        buffers/tensor.cc

        cpputils/logger.cc

        env/cart_pole.cc
        env/base.cc

        logging/client/base.cc
        logging/client/ema.cc
        logging/client/tensorboard.cc
        
        policies/categorical.cc
        policies/beta.cc
        policies/dirchlet.cc
        policies/normal.cc
        policies/gamma.cc

            policies/constraints/stack.cc
            policies/constraints/categorical_mask.cc
            policies/constraints/box.cc
            policies/constraints/empty.cc

        simulators/cart_pole.cc

        utils/n_step_collector.cc

        torchutils.cc
)

add_subdirectory(agents)
