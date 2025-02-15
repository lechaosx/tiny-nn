cmake_minimum_required(VERSION 3.30)

project(NN LANGUAGES CXX)

include(CTest)

add_subdirectory(lib)
add_subdirectory(tools)

if(BUILD_TESTING)
	add_subdirectory(tests)
endif()
