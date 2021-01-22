cmake_minimum_required(VERSION 3.13.0)
project(pybmixcpp)

set (CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -msse2 -funroll-loops -ftree-vectorize -fopenmp")
set(COMPILE_OPTIONS -D_REENTRANT -fPIC)
set(CMAKE_BUILD_TYPE Release)


SET(SOURCE_DIR "pybmix/core/pybmixcpp")
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

target_include_directories(pybmixcpp PUBLIC ${BAYESMIX_INCLUDE_PATHS})
target_compile_definitions(pybmixcpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(pybmixcpp PUBLIC bayesmixlib pthread protobuf)
target_compile_options(pybmixcpp PUBLIC ${COMPILE_OPTIONS})
