find_package(Protobuf REQUIRED)

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

if (NOT TARGET cppzmq)
    set(CPPZMQ_BUILD_TESTS OFF)
    add_subdirectory(cppzmq)
endif()
target_link_libraries(rl PUBLIC cppzmq)

if (NOT TARGET spdlog::spdlog)
    add_subdirectory(spdlog)
    set_property(TARGET spdlog PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
target_link_libraries(rl PUBLIC spdlog::spdlog)

if (NOT TARGET tensorboard_logger)
    add_subdirectory(tensorboard_logger)
endif()
target_link_libraries(rl PUBLIC tensorboard_logger)
