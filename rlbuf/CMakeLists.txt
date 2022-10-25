find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_program(gRPC_CPP_PLUGIN grpc_cpp_plugin)

message(STATUS ${gRPC_CPP_PLUGIN})

add_library(rlbuf SHARED)
target_link_libraries(rlbuf PUBLIC protobuf::libprotobuf)
target_link_libraries(rlbuf PUBLIC gRPC::grpc++)

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

        target_sources(rlbuf PRIVATE ${ITEM}.pb.cc)
        target_sources(rlbuf PRIVATE ${ITEM}.grpc.pb.cc)
    endforeach()

    target_include_directories(rlbuf PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
    
endfunction()

target_link_libraries(rl PUBLIC rlbuf)

build_protos(
    rlbuf/env/remote/cart_pole
    rlbuf/env/remote/lunar_lander
    rlbuf/env/remote/continuous_lunar_lander
)

