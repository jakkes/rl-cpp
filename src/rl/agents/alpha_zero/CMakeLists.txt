target_sources(
    rl
    PRIVATE
        ./mcts.cc
        ./modules/base.cc
        
        ./trainer_impl/self_play_worker.cc
)
