function(rl_add_test_target testname source)
    add_executable(rl-test-${testname} ${source})
    set_property(TARGET rl-test-${testname} PROPERTY CXX_STANDARD 20)
    target_link_libraries(rl-test-${testname} PUBLIC gtest gmock rl::rl)
    add_test(test-${testname} rl-test-${testname})
endfunction()

function(rl_append_test testname sources)
    target_sources(rl-test-${testname} PUBLIC ${sources})
endfunction()

rl_add_test_target(env test_env.cc)
rl_append_test(env env/test_cart_pole.cc)
rl_append_test(env env/test_sim_wrapper.cc)

rl_add_test_target(logging test_logging.cc)
rl_append_test(logging logging/test_ema.cc)

rl_add_test_target(policies test_policies.cc)
rl_append_test(policies policies/test_categorical.cc)
rl_append_test(policies policies/test_beta.cc)
rl_append_test(policies policies/test_normal.cc)
rl_append_test(policies policies/test_gamma.cc)
rl_append_test(policies policies/test_dirchlet.cc)

rl_append_test(policies policies/constraints/test_categorical_mask.cc)
rl_append_test(policies policies/constraints/test_box.cc)
rl_append_test(policies policies/constraints/test_type_casting.cc)

rl_add_test_target(cpputils test_cpputils.cc)
rl_append_test(cpputils cpputils/test_slice.cc)
rl_append_test(cpputils cpputils/test_metronome.cc)

rl_add_test_target(buffers test_buffers.cc)
rl_append_test(buffers buffers/test_tensor.cc)
rl_append_test(buffers buffers/test_tensor_and_object.cc)

rl_add_test_target(torchutils test_torchutils.cc)
rl_append_test(torchutils torchutils/test_execution_unit.cc)
rl_append_test(torchutils torchutils/test_gradient_norm.cc)
rl_append_test(torchutils torchutils/test_scale_gradients.cc)

rl_add_test_target(agents test_agents.cc)
rl_append_test(agents agents/alpha_zero/test_mcts.cc)
rl_append_test(agents agents/dqn/policies/test_uniform.cc)
rl_append_test(agents agents/dqn/value_parsers/test_estimated_mean.cc)
rl_append_test(agents agents/dqn/value_parsers/test_distributional.cc)
rl_append_test(agents agents/utils/test_distributional_loss.cc)

rl_add_test_target(simulators test_simulators.cc)
rl_append_test(simulators simulators/test_cart_pole.cc)
rl_append_test(simulators simulators/test_combinatorial.cc)

rl_add_test_target(utils test_utils.cc)
rl_append_test(utils utils/reward/test_n_step_collector.cc)
rl_append_test(utils utils/reward/test_backpropagate.cc)
