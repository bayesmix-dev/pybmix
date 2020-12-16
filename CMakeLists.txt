cmake_minimum_required(VERSION 3.13.0)
project(pybmix)

SET(SOURCE_DIR "pybmix/src/")
# Tell cmake that headers are in alse in source_dir
include_directories(${SOURCE_DIR})


# Generate python module
add_subdirectory(lib/pybind11)
pybind11_add_module(pybmix ${SOURCES} "${SOURCE_DIR}/example.cpp")
target_compile_definitions(pybmix PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})