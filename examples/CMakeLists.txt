add_subdirectory(dqn)
add_subdirectory(ppo)
add_subdirectory(sac)
add_subdirectory(alpha_zero)

add_executable(cudagraph cudagraph.cc)
target_link_libraries(cudagraph PRIVATE rl)
