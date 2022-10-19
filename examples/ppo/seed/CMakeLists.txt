add_executable(ppo-seed-cartpole cartpole.cc)
target_link_libraries(ppo-seed-cartpole PRIVATE rl)

add_executable(ppo-seed-continuous-cartpole continuous_cartpole.cc)
target_link_libraries(ppo-seed-continuous-cartpole PRIVATE rl)
