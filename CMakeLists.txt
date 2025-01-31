cmake_minimum_required(VERSION 3.31)

project(NN LANGUAGES CXX)

find_package(Eigen3 REQUIRED)

add_executable(main)

target_sources(main PRIVATE
	activation.h
	main.cpp
	idx.h
)

target_link_libraries(main PRIVATE Eigen3::Eigen)

target_compile_features(main PRIVATE cxx_std_26)