cmake_minimum_required(VERSION 3.30)

find_package(godot-cpp)

add_library(neural-network MODULE)

file(GLOB SOURCES CONFIGURE_DEPENDS *)

target_sources(neural-network PRIVATE ${SOURCES})

target_link_libraries(neural-network PRIVATE godot-cpp::godot-cpp nn)