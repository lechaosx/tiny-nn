cmake_minimum_required(VERSION 3.30)

project(NN LANGUAGES CXX)

include(CTest)

include(FetchContent)

FetchContent_Declare(
        GDExtension
        GIT_REPOSITORY https://github.com/godotengine/godot-cpp.git
        GIT_TAG godot-4.4.1-stable
)

FetchContent_MakeAvailable(GDExtension)

add_subdirectory(libnn)
add_subdirectory(tools)
add_subdirectory(gdextension)
add_subdirectory(godot)

