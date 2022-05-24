cmake_minimum_required(VERSION 3.13.0)
project(pybmix)

SET(SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/pybmixlib/bayesmix")

option(DISABLE_TESTS ON)
option(DISABLE_BENCHMARKS ON)
option(DISABLE_DOCS ON)
option(DISABLE_PLOTS ON)
option(DISABLE_EXAMPLES ON)
option(BUILD_RUN OFF)

add_subdirectory(${SOURCE_DIR})
include_directories(${SOURCE_DIR})
SET_SOURCE_FILES_PROPERTIES(${PROTO_HEADERS} ${PROTO_SOURCES} PROPERTIES GENERATED TRUE)

message ("*** PROTOC "  protobuf::protoc)
message ("*** Protobuf_VERSION "  ${Protobuf_VERSION})
message ("*** Protobuf_PROTOC_LIBRARIES "  ${Protobuf_PROTOC_LIBRARIES})



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
    COMMAND /usr/local/bin/protoc "--proto_path=${SOURCE_DIR}/src/proto"
    ${PROTO_DIRS} "--python_out=${PY_PROTO_DIR}" ${PROTO_FILE}
    DEPENDS ${PROTO_FILE} protobuf::protoc
    COMMENT "Generate Python protocol buffer for ${PROTO_FILE}"
    VERBATIM)
    list(APPEND PROTO_PYS ${PROTO_PY})
endforeach()

SET_SOURCE_FILES_PROPERTIES(${PROTO_PYS} PROPERTIES GENERATED TRUE)

add_custom_target(generate_protos ALL DEPENDS ${PROTO_PYS})
add_custom_target(two_to_three ALL COMMAND ${CMAKE_CURRENT_LIST_DIR}/convert_proto.sh DEPENDS generate_protos)
