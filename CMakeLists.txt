cmake_minimum_required(VERSION 3.30)

project(NN LANGUAGES CXX)

find_package(Eigen3 REQUIRED)

add_executable(main)
target_sources(main PRIVATE main.cpp activation.h loss.h)
target_link_libraries(main PRIVATE Eigen3::Eigen)
target_compile_features(main PRIVATE cxx_std_23)