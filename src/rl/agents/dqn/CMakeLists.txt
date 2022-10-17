target_sources(
    rl
    PRIVATE
        modules/distributional.cc
        modules/dqn.cc

        policies/epsilon_greedy.cc

        trainers/basic.cc
        trainers/seed.cc
        trainers/seed_impl/trainer.cc

        utils/hindsight_replay.cc
)
