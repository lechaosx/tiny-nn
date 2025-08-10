cmake_minimum_required(VERSION 3.30)

find_package(godot-cpp)

if(EMSCRIPTEN)
	add_executable(neural-network)
	target_compile_options(neural-network PRIVATE -fPIC)
	target_link_options(neural-network PRIVATE -s SIDE_MODULE=1)
	set_target_properties(neural-network PROPERTIES SUFFIX ".wasm")
else()
	add_library(neural-network MODULE)
endif()

file(GLOB SOURCES CONFIGURE_DEPENDS *)

target_sources(neural-network PRIVATE ${SOURCES})

target_link_libraries(neural-network PRIVATE godot-cpp::godot-cpp nn)