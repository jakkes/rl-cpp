add_executable(ppo-cartpole ppo_cartpole.cc)
target_link_libraries(ppo-cartpole PRIVATE rl)

add_executable(ppo-continuous-cartpole ppo_continuous_cartpole.cc)
target_link_libraries(ppo-continuous-cartpole PRIVATE rl)
