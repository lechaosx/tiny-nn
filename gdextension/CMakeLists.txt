cmake_minimum_required(VERSION 3.30)

add_library(neural-network SHARED)

file(GLOB SOURCES CONFIGURE_DEPENDS *)

target_sources(neural-network PRIVATE ${SOURCES})

target_link_libraries(neural-network PRIVATE godot-cpp nn)

if(CMAKE_SYSTEM_NAME STREQUAL Emscripten)
	target_link_options(neural-network PRIVATE -sUSE_PTHREADS=1)
endif()