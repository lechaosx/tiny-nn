cmake_minimum_required(VERSION 3.30)

project(NN LANGUAGES CXX)

include(CTest)

add_subdirectory(libnn)
add_subdirectory(tools)
add_subdirectory(gdextension)
add_subdirectory(godot)

