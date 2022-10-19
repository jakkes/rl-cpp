add_executable(ppo-basic-cartpole cartpole.cc)
target_link_libraries(ppo-basic-cartpole PRIVATE rl)

add_executable(ppo-basic-continuous-cartpole continuous_cartpole.cc)
target_link_libraries(ppo-basic-continuous-cartpole PRIVATE rl)
