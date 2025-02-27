cmake_minimum_required(VERSION 3.30)

find_package(godot-cpp)

add_library(neural-network MODULE)

file(GLOB SOURCES CONFIGURE_DEPENDS *)

target_sources(neural-network PRIVATE ${SOURCES})

target_link_libraries(neural-network PRIVATE godot-cpp::godot-cpp nn)

set_target_properties(neural-network PROPERTIES OUTPUT_NAME neural_network.$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>.$<LOWER_CASE:$<CONFIG>>.${CMAKE_SYSTEM_PROCESSOR})

install(TARGETS neural-network DESTINATION . RUNTIME_DEPENDENCIES COMPONENT gdextension)