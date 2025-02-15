cmake_minimum_required(VERSION 3.30)

find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(train)

file(GLOB SOURCES CONFIGURE_DEPENDS *)

target_sources(train PRIVATE ${SOURCES})

target_link_libraries(train PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)

target_compile_features(train PRIVATE cxx_std_26)