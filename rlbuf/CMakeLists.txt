find_package(Protobuf REQUIRED)

add_library(rlbuf SHARED)

function(build_protos)
    foreach(ITEM IN LISTS ARGN)
        execute_process (
            COMMAND ${Protobuf_PROTOC_EXECUTABLE}
                -I ${CMAKE_CURRENT_LIST_DIR}
                --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
                --plugin=gRPC::cpp_plugin
                ${ITEM}.proto
        )

        target_sources(rlbuf PRIVATE ${ITEM}.pb.cc)
        target_sources(rlbuf PRIVATE ${ITEM}.pb.grpc.cc)
    endforeach()

    target_include_directories(rlbuf PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
    
endfunction()


build_protos(rlbuf/env/cart_pole)
