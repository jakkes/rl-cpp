target_sources(
    rl
    PRIVATE
        ./mcts.cc
        ./modules/base.cc
        ./modules/fixed_value_support.cc
        
        ./trainer.cc
        ./trainer_impl/self_play_worker.cc
        ./trainer_impl/trainer.cc
)
