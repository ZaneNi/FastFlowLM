cmake_minimum_required(VERSION 3.22)
project(image_processor_project LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/image_preprocess/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(XRT_INCLUDE_DIR C:/dev/XRT/src/runtime_src/core/include CACHE PATH   "Where XRT headers live")
set(XRT_LIB_DIR     C:/dev/xrtNPUfromDLL     CACHE PATH   "Where XRT libs live")


add_executable(image_processor 
    ${CMAKE_SOURCE_DIR}/test.cpp
    ${CMAKE_SOURCE_DIR}/../../common/image/image_reader.cpp
)

target_include_directories(image_processor
    PUBLIC
        ${CMAKE_SOURCE_DIR}/../../include
        C:/dev/vcpkg/installed/x64-windows/include
    ${XRT_INCLUDE_DIR}
    C:/Program\ Files/boost/boost_1_88_0
)

target_compile_definitions(image_processor
    PUBLIC
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
        _CRT_NONSTDC_NO_DEPRECATE
        DISABLE_ABI_CHECK=1
)

target_link_directories(image_processor
    PUBLIC C:/dev/vcpkg/installed/x64-windows/lib
    ${XRT_LIB_DIR}
)

target_link_libraries(image_processor
    PUBLIC avformat
    PUBLIC avcodec
    PUBLIC avutil
    PUBLIC swscale
    PUBLIC zlib
    PUBLIC xrt_coreutil
)

# Extra compiler flags
if(MSVC)
    target_compile_options(image_processor PRIVATE /W4)
else()
    target_compile_options(image_processor PRIVATE -Wall -Wextra)
endif()


target_compile_definitions(image_processor
    PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
        _CRT_NONSTDC_NO_DEPRECATE
)