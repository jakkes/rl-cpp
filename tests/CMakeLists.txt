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

rl_add_test_target(torch test_torch.cc)
rl_append_test(torch torch/test_array_ref_slice.cc)

rl_add_test_target(policies test_policies.cc)
rl_append_test(
    policies
        policies/test_categorical.cc
        policies/constraints/test_cat.cc
        policies/constraints/test_categorical_mask.cc
)
