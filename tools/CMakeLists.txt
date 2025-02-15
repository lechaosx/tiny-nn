cmake_minimum_required(VERSION 3.30)

add_executable(train)

file(GLOB SOURCES CONFIGURE_DEPENDS *)

target_sources(train PRIVATE ${SOURCES})

target_link_libraries(train PRIVATE nn)

target_compile_features(train PRIVATE cxx_std_26)