cmake_minimum_required(VERSION 3.13.0)
project(pybmixcpp)

SET(SOURCE_DIR "pybmixcpp/src")
# Tell cmake that headers are in alse in source_dir
add_subdirectory(${SOURCE_DIR}/bayesmix)
add_subdirectory(${SOURCE_DIR})
include_directories(${SOURCE_DIR})
set(COMPILE_OPTIONS -D_REENTRANT -fPIC)

message("**** BAYESMIX_INCLUDE_PATHS ${BAYESMIX_INCLUDE_PATHS}")

# Generate python module
add_subdirectory(lib/pybind11)
pybind11_add_module(pybmixcpp ${SOURCES} 
    "${SOURCE_DIR}/example.cpp" 
    "${SOURCE_DIR}/algorithm_wrapper.cpp")
target_include_directories(pybmixcpp PUBLIC ${BAYESMIX_INCLUDE_PATHS})
target_compile_definitions(pybmixcpp PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(pybmixcpp PUBLIC bayesmixlib tbb pthread protobuf)
target_compile_options(pybmixcpp PUBLIC ${COMPILE_OPTIONS})


# target_link_libraries(pybmix PRIVATE bayesmixlib tbb pthread protobuf)
# target_compile_options(pybmix PUBLIC -D_REENTRANT -fPIC)