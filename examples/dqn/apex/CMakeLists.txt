if (RL_BUILD_REMOTE_ENVS)
    add_executable(distributional-dqn-apex-cartpole distributional_cartpole.cc)
    target_link_libraries(distributional-dqn-apex-cartpole PRIVATE rl::rl)

    add_executable(distributional-dqn-apex-lunar-lander distributional_lunar_lander.cc)
    target_link_libraries(distributional-dqn-apex-lunar-lander PRIVATE rl::rl rl::remote-env)
endif()
