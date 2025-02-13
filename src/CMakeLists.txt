cmake_minimum_required(VERSION 3.30)

find_package(Eigen3 REQUIRED)

add_executable(train)

target_sources(train PRIVATE
	activation.h
	main.cpp
	idx.h
)

target_link_libraries(train PRIVATE Eigen3::Eigen)

target_compile_features(train PRIVATE cxx_std_26)