add_executable(sac-basic-continuous-cartpole continuous_cartpole.cc)
target_link_libraries(sac-basic-continuous-cartpole PRIVATE rl)

add_executable(sac-basic-continuous-lunar-lander continuous_lunar_lander.cc)
target_link_libraries(sac-basic-continuous-lunar-lander PRIVATE rl)