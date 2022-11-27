find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_program(gRPC_CPP_PLUGIN grpc_cpp_plugin)

add_library(rlbuf-remote-env SHARED)
add_library(rlbuf::remote-env ALIAS rlbuf-remote-env)

target_link_libraries(rlbuf-remote-env PUBLIC protobuf::libprotobuf)
target_link_libraries(rlbuf-remote-env PUBLIC gRPC::grpc++)

function(build_protos)
    foreach(ITEM IN LISTS ARGN)
        execute_process (
            COMMAND ${Protobuf_PROTOC_EXECUTABLE}
                -I ${CMAKE_CURRENT_LIST_DIR}
                --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
                --grpc_out=${CMAKE_CURRENT_BINARY_DIR}
                --plugin=protoc-gen-grpc=${gRPC_CPP_PLUGIN}
                ${ITEM}.proto
        )

        target_sources(rlbuf-remote-env PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${ITEM}.pb.cc)
        target_sources(rlbuf-remote-env PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${ITEM}.grpc.pb.cc)
    endforeach()

    target_include_directories(rlbuf-remote-env PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
    
endfunction()

target_link_libraries(rl-remote-env PUBLIC rlbuf-remote-env)

build_protos(
    rlbuf/remote_env/cart_pole
    rlbuf/remote_env/lunar_lander
)
