add_executable(ppo-cartpole ppo_cartpole.cc)
target_link_libraries(ppo-cartpole PRIVATE rl)

add_executable(seedppo-cartpole seedppo_cartpole.cc)
target_link_libraries(seedppo-cartpole PRIVATE rl)

add_executable(seedppo-continuous-cartpole seedppo_continuous_cartpole.cc)
target_link_libraries(seedppo-continuous-cartpole PRIVATE rl)

add_executable(ppo-continuous-cartpole ppo_continuous_cartpole.cc)
target_link_libraries(ppo-continuous-cartpole PRIVATE rl)
