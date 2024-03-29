set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set(MAIN_FILE ON)
else()
    set(MAIN_FILE OFF)
endif()

if (NOT DEFINED CAFFE2_USE_CUDNN)
    set(CAFFE2_USE_CUDNN ON)
endif()

cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES "75")

project(rl LANGUAGES CXX)

include(cmake/ccache.cmake)
include(cmake/submodule.cmake)

option(RL_BUILD_TESTS "build tests" ${MAIN_FILE})
option(RL_BUILD_EXAMPLES "build examples" ${MAIN_FILE})
option(RL_BUILD_REMOTE_ENVS "remote envs, requires gRPC" ON)

add_library(rl SHARED)
add_library(rl::rl ALIAS rl)
set_property(TARGET rl PROPERTY CXX_STANDARD 20)

target_include_directories(rl PUBLIC include)
target_include_directories(rl PRIVATE src/rl)

find_package(Torch REQUIRED)
target_link_libraries(rl PUBLIC torch)

add_subdirectory(libs)
add_subdirectory(src/rl)

if (RL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if (RL_BUILD_REMOTE_ENVS)
    add_subdirectory(remote_env)
endif()

if (RL_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
