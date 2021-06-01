cmake_minimum_required(VERSION 3.13.0)
project(pybmixcpp)

set (CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -msse2 -funroll-loops -ftree-vectorize -fopenmp")
set(COMPILE_OPTIONS -D_REENTRANT -fPIC)
set(CMAKE_BUILD_TYPE Release)


SET(SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/pybmix/core/pybmixcpp")
add_subdirectory(${SOURCE_DIR}/bayesmix)
include_directories(${SOURCE_DIR})
SET_SOURCE_FILES_PROPERTIES(${PROTO_HEADERS} ${PROTO_SOURCES} PROPERTIES GENERATED TRUE)

message("**** BAYESMIX_INCLUDE_PATHS ${BAYESMIX_INCLUDE_PATHS}")


# Generate python module
add_subdirectory(lib/pybind11)
pybind11_add_module(pybmixcpp ${SOURCES}
    "${SOURCE_DIR}/module.cpp"
    "${SOURCE_DIR}/algorithm_wrapper.hpp"
    "${SOURCE_DIR}/algorithm_wrapper.cpp"
    "${SOURCE_DIR}/serialized_collector.hpp"
    "${SOURCE_DIR}/serialized_collector.cpp"
    ${PROTO_HEADERS} ${PROTO_SOURCES})

# generate Python's proto classes
set(PY_PROTO_DIR "${CMAKE_CURRENT_LIST_DIR}/pybmix/proto")
set(PROTO_PYS)
message("ProtoFiles: ${ProtoFiles}")
foreach(PROTO_FILE IN LISTS ProtoFiles)
  message(STATUS "GENERATING PYTHON protoc proto(cc): ${PROTO_FILE}")
  get_filename_component(PROTO_DIR ${PROTO_FILE} DIRECTORY)
  get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)
  set(PROTO_PY ${PY_PROTO_DIR}/${PROTO_NAME}_pb2.py)
  add_custom_command(
    OUTPUT ${PROTO_PY}
    COMMAND protobuf::protoc "--proto_path=${SOURCE_DIR}/bayesmix/proto"
    ${PROTO_DIRS} "--python_out=${PY_PROTO_DIR}" ${PROTO_FILE}
    DEPENDS ${PROTO_FILE} protobuf::protoc
    COMMENT "Generate Python protocol buffer for ${PROTO_FILE}"
    VERBATIM)
    list(APPEND PROTO_PYS ${PROTO_PY})
endforeach()

set(PYBMIX_LINK_LIBRARIES 
  pthread 
  protobuf::libprotobuf
  tbb
)

target_include_directories(pybmixcpp PUBLIC ${BAYESMIX_INCLUDE_PATHS})
target_compile_definitions(pybmixcpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(pybmixcpp PUBLIC "-L${CMAKE_CURRENT_LIST_DIR}/pybmix/core/pybmixcpp/bayesmix/lib/math/lib/tbb/" bayesmixlib ${PYBMIX_LINK_LIBRARIES})
target_compile_options(pybmixcpp PUBLIC ${BAYESMIX_COMPILE_OPTIONS})

add_custom_target(genterate_protos ALL DEPENDS ${PROTO_PYS})

