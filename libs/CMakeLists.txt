if (NOT TARGET gtest AND RL_BUILD_TESTS)
    add_subdirectory(googletest)
endif()
