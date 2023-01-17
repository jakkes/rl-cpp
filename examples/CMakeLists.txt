add_subdirectory(dqn)
add_subdirectory(ppo)
add_subdirectory(sac)
add_subdirectory(alpha_zero)

add_executable(cublas_test cublas_test.cc)
target_link_libraries(cublas_test PRIVATE rl)
