cmake_minimum_required(VERSION 3.30)

project(PackageTest CXX)

find_package(godot-cpp REQUIRED)

add_library(example MODULE src/example.cpp)

target_link_libraries(example godot-cpp::godot-cpp)
