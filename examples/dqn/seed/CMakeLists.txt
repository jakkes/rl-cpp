if (RL_BUILD_REMOTE_ENVS)
    add_executable(distributional-dqn-seed-cartpole distributional_cartpole.cc)
    target_link_libraries(distributional-dqn-seed-cartpole PRIVATE rl::rl rl::remote-env)

    add_executable(distributional-dqn-seed-lunar-lander distributional_lunar_lander.cc)
    target_link_libraries(distributional-dqn-seed-lunar-lander PRIVATE rl::rl rl::remote-env)
endif()
