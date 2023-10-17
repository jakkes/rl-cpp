target_sources(
    rl
    PRIVATE
        value_parsers/estimated_mean.cc
        value_parsers/distributional.cc

        policies/epsilon_greedy.cc

        trainers/basic.cc
        trainers/seed.cc
        trainers/seed_impl/trainer.cc
        trainers/seed_impl/inferer.cc
        trainers/seed_impl/env_thread.cc
        trainers/seed_impl/transition_collector.cc
        
        trainers/apex.cc
        trainers/apex_impl/trainer.cc
        trainers/apex_impl/worker.cc

        utils/hindsight_replay.cc
)
