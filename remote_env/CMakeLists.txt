add_library(rl-remote-env SHARED)
add_library(rl::remote-env ALIAS rl-remote-env)
set_property(TARGET rl-remote-env PROPERTY CXX_STANDARD 20)

target_include_directories(rl-remote-env PUBLIC include)

add_subdirectory(rlbuf)
add_subdirectory(src)

if (RL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

target_link_libraries(rl-remote-env PUBLIC rl)
