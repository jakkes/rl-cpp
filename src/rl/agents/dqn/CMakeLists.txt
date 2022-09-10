target_sources(
    rl
    PRIVATE
        modules/distributional.cc
        modules/dqn.cc
        modules/base.cc

        policies/epsilon_greedy.cc

        trainers/basic.cc
)
