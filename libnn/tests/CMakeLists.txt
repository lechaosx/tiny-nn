cmake_minimum_required(VERSION 3.30)

find_package(GTest REQUIRED)

include(GoogleTest)

add_executable(nn_tests)

file(GLOB SOURCES CONFIGURE_DEPENDS *)

target_sources(nn_tests PRIVATE ${SOURCES})

target_link_libraries(nn_tests PRIVATE GTest::gtest_main nn)

gtest_discover_tests(nn_tests)