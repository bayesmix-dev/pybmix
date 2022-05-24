cmake_minimum_required(VERSION 3.13.0)
project(pybmixcpp)

set (CMAKE_CXX_STANDARD 14)
set(COMPILE_OPTIONS -D_REENTRANT -fPIC)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -ftree-vectorize")
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)


SET(SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/pybmix/core/pybmixcpp")
add_subdirectory(${SOURCE_DIR}/bayesmix)
include_directories(${SOURCE_DIR})
SET_SOURCE_FILES_PROPERTIES(${PROTO_HEADERS} ${PROTO_SOURCES} PROPERTIES GENERATED TRUE)

# Generate python module
add_subdirectory(lib/pybind11)
pybind11_add_module(pybmixcpp ${SOURCES}
    "${SOURCE_DIR}/bayesmix/module.cpp"
    "${SOURCE_DIR}/bayesmix/algorithm_wrapper.hpp"
    "${SOURCE_DIR}/bayesmix/algorithm_wrapper.cpp"
    "${SOURCE_DIR}/bayesmix/serialized_collector.hpp"
    "${SOURCE_DIR}/bayesmix/serialized_collector.cpp"
    ${PROTO_HEADERS} ${PROTO_SOURCES})

# generate Python's proto classes
set(PY_PROTO_DIR "${CMAKE_CURRENT_LIST_DIR}/pybmix/proto")
set(PROTO_PYS)
message("ProtoFiles: ${ProtoFiles}")
foreach(PROTO_FILE IN LISTS ProtoFiles)
  get_filename_component(PROTO_DIR ${PROTO_FILE} DIRECTORY)
  get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)
  set(PROTO_PY ${PY_PROTO_DIR}/${PROTO_NAME}_pb2.py)
  message(STATUS "GENERATING PYTHON protoc proto(cc): ${PROTO_FILE} --> ${PROTO_PY}")
  add_custom_command(
    OUTPUT ${PROTO_PY}
    COMMAND protobuf::protoc "--proto_path=${SOURCE_DIR}/bayesmix/bayesmix/src/proto"
    ${PROTO_DIRS} "--python_out=${PY_PROTO_DIR}" ${PROTO_FILE}
    DEPENDS ${PROTO_FILE} protobuf::protoc
    COMMENT "Generate Python protocol buffer for ${PROTO_FILE}"
    VERBATIM)
    list(APPEND PROTO_PYS ${PROTO_PY})
endforeach()

find_package (TBB EXACT PATHS ${TBB_ROOT})
find_package(OpenMP REQUIRED)

message("INCLUDE_PATHS" ${INCLUDE_PATHS})

target_include_directories(pybmixcpp PUBLIC ${BAYESMIX_INCLUDE_PATHS})
target_compile_definitions(pybmixcpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(pybmixcpp PUBLIC bayesmixlib ${BAYESMIX_LINK_LIBRARIES})
target_compile_options(pybmixcpp PUBLIC ${BAYESMIX_COMPILE_OPTIONS})

add_custom_target(generate_protos ALL DEPENDS ${PROTO_PYS})
add_custom_target(two_to_three ALL COMMAND ${CMAKE_CURRENT_LIST_DIR}/convert_proto.sh DEPENDS generate_protos)