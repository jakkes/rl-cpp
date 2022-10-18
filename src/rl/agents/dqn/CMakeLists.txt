target_sources(
    rl
    PRIVATE
        modules/distributional.cc
        modules/dqn.cc

        policies/epsilon_greedy.cc

        trainers/basic.cc
        trainers/seed.cc
        trainers/seed_impl/trainer.cc
        trainers/seed_impl/inferer.cc
        trainers/seed_impl/env_thread.cc
        trainers/seed_impl/transition_collector.cc

        utils/hindsight_replay.cc
)
