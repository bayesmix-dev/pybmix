cmake_minimum_required(VERSION 3.13.0)
project(pybmix)

SET(SOURCE_DIR "pybmix/src")
# Tell cmake that headers are in alse in source_dir
add_subdirectory(${SOURCE_DIR}/bayesmix)
add_subdirectory(${SOURCE_DIR})
include_directories(${SOURCE_DIR})
set(INCLUDE_PATHS
    ${SOURCE_DIR}
    ${SOURCE_DIR}/bayesmix/lib/math
    ${SOURCE_DIR}/bayesmix/lib/math/lib/eigen_3.3.9
    ${SOURCE_DIR}/bayesmix/lib/math/lib/boost_1.72.0
)
set(COMPILE_OPTIONS -D_REENTRANT -fPIC)

message("**** INCLUDE PATHS ${INCLUDE_PATHS}")

# Generate python module
add_subdirectory(lib/pybind11)
pybind11_add_module(pybmix ${SOURCES} 
    "${SOURCE_DIR}/example.cpp" 
    "${SOURCE_DIR}/algorithm_wrapper.cpp")
target_include_directories(pybmix PUBLIC ${INCLUDE_PATHS})
target_compile_definitions(pybmix PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(pybmix PUBLIC bayesmixlib tbb pthread protobuf)
target_compile_options(pybmix PUBLIC ${COMPILE_OPTIONS})


# target_link_libraries(pybmix PRIVATE bayesmixlib tbb pthread protobuf)
# target_compile_options(pybmix PUBLIC -D_REENTRANT -fPIC)