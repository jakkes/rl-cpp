function(rl_add_test_target testname source)
    add_executable(rl-test-${testname} ${source})
    target_link_libraries(rl-test-${testname} PUBLIC gtest rl::rl)
    add_test(test-${testname} rl-test-${testname})
endfunction()

function(rl_append_test testname source)
    target_sources(rl-test-${testname} PUBLIC ${source})
endfunction()

rl_add_test_target(env test_env.cc)
rl_append_test(env env/test_simple_env.cc)
