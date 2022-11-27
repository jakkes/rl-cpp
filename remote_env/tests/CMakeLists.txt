rl_add_test_target(test_env_remote test_env_remote.cc)
rl_append_test(test_env_remote ./env/remote/test_remote_cart_pole.cc)
rl_append_test(test_env_remote ./env/remote/test_remote_lunar_lander.cc)
