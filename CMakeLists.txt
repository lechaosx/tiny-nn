cmake_minimum_required(VERSION 3.30)

project(NN LANGUAGES CXX)

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

include(CTest)

include(FetchContent)

FetchContent_Declare(GDExtension GIT_REPOSITORY https://github.com/godotengine/godot-cpp.git GIT_TAG godot-4.5-stable)

FetchContent_MakeAvailable(GDExtension)

include(${godot-cpp_SOURCE_DIR}/cmake/emsdkHack.cmake)

add_subdirectory(libnn)
add_subdirectory(tools)
add_subdirectory(gdextension)
add_subdirectory(godot)

