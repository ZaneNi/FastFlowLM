cmake_minimum_required(VERSION 3.22)
project(q4nx VERSION 1.0.0 LANGUAGES CXX)

# Set build type to Release
set(CMAKE_BUILD_TYPE Release)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/q4nx/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/q4nx/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../../build/q4nx/lib)

# Force output directories to be absolute
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

# ———————————————————————————————————————————————
# User-tweakable XRT settings (override on cmake command-line)
# ———————————————————————————————————————————————
set(XRT_INCLUDE_DIR C:/dev/XRT/src/runtime_src/core/include CACHE PATH   "Where XRT headers live")
set(XRT_LIB_DIR     C:/dev/xrtNPUfromDLL     CACHE PATH   "Where XRT libs live")


# ———————————————————————————————————————————————
# Test executable target
# ———————————————————————————————————————————————
# Set test output directory
set(TEST_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/../../build/q4nx/test/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${TEST_OUTPUT_DIR})

# Create test executable with basic settings
add_executable(test_q4nx test.cpp ${HEADERS})

# Basic include directories for test
target_include_directories(test_q4nx PUBLIC
    ${CMAKE_SOURCE_DIR}/../../include
    ${CMAKE_SOURCE_DIR}
    ${XRT_INCLUDE_DIR}
    C:/Program\ Files/boost/boost_1_88_0
)

target_compile_definitions(test_q4nx PUBLIC
    DISABLE_ABI_CHECK=1
    USEAVX2=1
)

target_compile_options(test_q4nx PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2 /O2 /fp:fast>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-mavx2 -mfma -march=native -O2>
)

# Basic link settings for test
target_link_directories(test_q4nx PUBLIC
    ${CMAKE_SOURCE_DIR}/../../build/q4nx/lib
    ${XRT_LIB_DIR}
    C:/Program\ Files/boost/boost_1_88_0/lib64-msvc-14.3
)

target_link_libraries(test_q4nx PUBLIC
    q4_npu_eXpress
    xrt_coreutil
)

# Add test target
add_custom_target(test_q4nx_target
    DEPENDS test_q4nx
    COMMENT "Building test_q4nx executable"
) 