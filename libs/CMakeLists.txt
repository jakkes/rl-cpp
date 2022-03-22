if (NOT TARGET gtest AND RL_BUILD_TESTS)
    add_subdirectory(googletest)
endif()

target_include_directories(rl PUBLIC thread-pool)

if (NOT TARGET argparse)
    add_subdirectory(argparse)
endif()
target_link_libraries(rl PUBLIC argparse)

if (NOT TARGET thread-safe::collections)
    add_subdirectory(thread-safe-collections)
endif()
target_link_libraries(rl PUBLIC thread-safe::collections)
