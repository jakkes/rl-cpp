function(rl_add_test_target testname source)
    add_executable(rl-test-${testname} ${source})
    target_link_libraries(rl-test-${testname} PUBLIC gtest rl::rl)
    add_test(test-${testname} rl-test-${testname})
endfunction()

function(rl_append_test testname sources)
    target_sources(rl-test-${testname} PUBLIC ${sources})
endfunction()

rl_add_test_target(env test_env.cc)
rl_append_test(env env/test_simple_env.cc)

rl_add_test_target(policies test_policies.cc)
rl_append_test(policies policies/test_categorical.cc)
rl_append_test(policies policies/test_rejection_sampler.cc)
rl_append_test(policies policies/constraints/test_cat.cc)
rl_append_test(policies policies/constraints/test_categorical_mask.cc)

rl_add_test_target(cpputils test_cpputils.cc)
rl_append_test(
    cpputils
        cpputils/test_slice.cc
)

rl_add_test_target(buffers test_buffers.cc)
rl_append_test(buffers buffers/test_state_transition.cc)
rl_append_test(buffers buffers/test_tensor.cc)
