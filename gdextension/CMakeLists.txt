cmake_minimum_required(VERSION 3.30)

add_library(neural-network SHARED)

file(GLOB SOURCES CONFIGURE_DEPENDS *)

target_sources(neural-network PRIVATE ${SOURCES})

target_link_libraries(neural-network PRIVATE godot-cpp nn)