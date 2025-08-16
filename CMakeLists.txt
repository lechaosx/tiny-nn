cmake_minimum_required(VERSION 3.30)

project(NN LANGUAGES CXX)

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(EMSCRIPTEN)
    set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
    set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "-sSIDE_MODULE=1")
    set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "-sSIDE_MODULE=1")
    set(CMAKE_SHARED_LIBRARY_SUFFIX)
    set(CMAKE_STRIP FALSE)
    set(CMAKE_SYSTEM_PROCESSOR "wasm32")
endif()

include(CTest)

include(FetchContent)

FetchContent_Declare(GDExtension GIT_REPOSITORY https://github.com/godotengine/godot-cpp.git GIT_TAG godot-4.4.1-stable)

FetchContent_MakeAvailable(GDExtension)

add_subdirectory(libnn)
add_subdirectory(tools)
add_subdirectory(gdextension)
add_subdirectory(godot)

