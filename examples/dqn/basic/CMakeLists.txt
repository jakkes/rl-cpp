add_executable(dqn-basic-cartpole cartpole.cc)
target_link_libraries(dqn-basic-cartpole PRIVATE rl)

add_executable(distributional-dqn-basic-cartpole distributional_cartpole.cc)
target_link_libraries(distributional-dqn-basic-cartpole PRIVATE rl)
