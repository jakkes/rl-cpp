find_package(gRPC CONFIG REQUIRED)

target_include_directories(rl PUBLIC include/)

add_subdirectory(src)

if (RL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
