cmake_minimum_required(VERSION 3.30)

find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)

add_library(nn INTERFACE)

file(GLOB SOURCES CONFIGURE_DEPENDS *)

target_sources(nn INTERFACE ${SOURCES})

target_include_directories(nn INTERFACE .)

target_link_libraries(nn INTERFACE Eigen3::Eigen nlohmann_json::nlohmann_json)

target_compile_features(nn INTERFACE cxx_std_26)