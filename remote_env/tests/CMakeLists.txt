rl_add_test_target(remote-env test_remote_env.cc)
rl_append_test(remote-env ./remote_env/test_remote_cart_pole.cc)
rl_append_test(remote-env ./remote_env/test_remote_lunar_lander.cc)

target_link_libraries(rl-test-remote-env PRIVATE rl-remote-env)
