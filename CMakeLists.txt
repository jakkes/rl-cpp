if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set(MAIN_FILE ON)
else()
    set(MAIN_FILE OFF)
endif()

cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES "75")

project(rl LANGUAGES CXX)

option(RL_BUILD_TESTS "build tests" ${MAIN_FILE})

add_library(rl SHARED)
add_library(rl::rl ALIAS rl)
set_property(TARGET rl PROPERTY CXX_STANDARD 20)

target_include_directories(rl PUBLIC include)

find_package(Torch REQUIRED)
target_link_libraries(rl PUBLIC torch)

add_subdirectory(src)
add_subdirectory(libs)

if (RL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
