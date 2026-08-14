# CMakeLists.txt — cross-platform build for kimi-k3-lean
#
# Mirrors the Makefile. Use CMake if you prefer it, or if you're on Windows,
# macOS without make, or building with Visual Studio / Ninja / Xcode.
#
# Quick start:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ctest --test-dir build
#
# Custom prefix (defaults to /usr/local):
#   cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$HOME/.local
#   cmake --install build

cmake_minimum_required(VERSION 3.16)
project(kimi-k3-lean
    VERSION 0.6.8
    DESCRIPTION "Lean OpenAI-compatible server for Kimi K3 — disk-resident, CPU-only"
    LANGUAGES C)

# ---------------------------------------------------------------------------- options

option(K3_BUILD_CLI   "Build the k3 CLI binary"            ON)
option(K3_BUILD_LIB   "Build libk3 (static + shared)"       ON)
option(K3_BUILD_TESTS "Build unit tests"                    ON)
option(K3_INSTALL_HEADERS "Install headers (libk3.h)"        ON)

set(K3_VERSION "0.6.8" CACHE STRING "Engine version")

# ---------------------------------------------------------------------------- compile flags

# Sane defaults. Override with -DCMAKE_C_FLAGS=...
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)        # we use POSIX extensions (clock_gettime, etc.)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Match the Makefile: AVX2+FMA where supported, no AVX-512.
# Detect at configure time so we get the right code path on x86 and ARM.
include(CheckCCompilerFlag)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|i.86|x86")
    check_c_compiler_flag("-mavx2 -mfma" K3_HAVE_AVX2)
    if(K3_HAVE_AVX2)
        add_compile_options(-mavx2 -mfma -ffp-contract=off)
        message(STATUS "AVX2+FMA enabled (x86_64)")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    check_c_compiler_flag("-march=armv8.2-a+fp16+dotprod+i8mm" K3_HAVE_ARM82)
    if(K3_HAVE_ARM82)
        add_compile_options(-march=armv8.2-a+fp16+dotprod+i8mm)
        message(STATUS "ARMv8.2 + dotprod + i8mm enabled (aarch64)")
    else()
        add_compile_options(-march=armv8-a)
        message(STATUS "ARMv8 baseline (aarch64, no SVE)")
    endif()
endif()

# Threading. -fopenmp on Linux/macOS; on Windows, MSVC doesn't have OMP by default.
find_package(Threads REQUIRED)

# libm is implicit on most POSIX; on Windows it's separate.
if(WIN32)
    find_library(M_LIB m REQUIRED)
    set(PLATFORM_LIBS m)
else()
    set(PLATFORM_LIBS m)
endif()

# ---------------------------------------------------------------------------- source lists

set(K3_ENGINE_SRC
    src/core/k3_ops.c
    src/io/k3_st.c
    src/io/k3_load.c
    src/io/k3_trunk.c
    src/cache/k3_cache.c
    src/model/k3_bind.c
)

set(K3_LIB_SRC
    src/lib/k3_engine.c
    src/lib/k3_api.c
)

set(K3_CLI_SRC
    src/cli/k3_run.c
)

# All source files, for IDE convenience (one target with everything).
set(K3_ALL_SRC
    ${K3_ENGINE_SRC}
    ${K3_LIB_SRC}
    ${K3_CLI_SRC}
)

# ---------------------------------------------------------------------------- library target

add_library(k3_engine STATIC ${K3_ENGINE_SRC})
target_include_directories(k3_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/k3
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cache
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer
)
target_link_libraries(k3_engine PUBLIC ${PLATFORM_LIBS} Threads::Threads)

# The combined library that the Python server uses via ctypes.
if(K3_BUILD_LIB)
    add_library(k3 SHARED ${K3_LIB_SRC})
    target_include_directories(k3 PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/libk3
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(k3 PUBLIC k3_engine)
    set_target_properties(k3 PROPERTIES
        OUTPUT_NAME k3
        VERSION ${K3_VERSION}
        SOVERSION ${K3_VERSION}
    )

    # Static variant for embedding.
    add_library(k3_static STATIC ${K3_LIB_SRC})
    target_include_directories(k3_static PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include/libk3
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(k3_static PUBLIC k3_engine)
    set_target_properties(k3_static PROPERTIES OUTPUT_NAME k3_static)
endif()

# ---------------------------------------------------------------------------- CLI target

if(K3_BUILD_CLI)
    add_executable(k3 ${K3_CLI_SRC})
    target_link_libraries(k3 PRIVATE k3_engine k3_static)
    target_include_directories(k3 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/libk3
    )
endif()

# ---------------------------------------------------------------------------- tests

if(K3_BUILD_TESTS)
    enable_testing()

    # Engine unit tests (the article's ctest target).
    file(GLOB K3_TEST_SOURCES tests/unit/*.c)
    foreach(src ${K3_TEST_SOURCES})
        get_filename_component(name ${src} NAME_WE)
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE k3_engine)
        target_include_directories(${name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/include/k3
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
        )
        add_test(NAME ${name} COMMAND ${name})
        set_tests_properties(${name} PROPERTIES
            ENVIRONMENT "FIXTURES=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
        )
    endforeach()
endif()

# ---------------------------------------------------------------------------- install

include(GNUInstallDirs)

if(K3_INSTALL_HEADERS)
    install(FILES include/libk3/libk3.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libk3)
endif()

if(K3_BUILD_LIB)
    install(TARGETS k3 k3_static
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}    # for Windows .dll
    )
endif()

if(K3_BUILD_CLI)
    install(TARGETS k3 RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# ---------------------------------------------------------------------------- summary

message(STATUS "")
message(STATUS "kimi-k3-lean ${K3_VERSION} configuration:")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build CLI:         ${K3_BUILD_CLI}")
message(STATUS "  Build library:     ${K3_BUILD_LIB}")
message(STATUS "  Build tests:       ${K3_BUILD_TESTS}")
message(STATUS "  Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Install headers:   ${K3_INSTALL_HEADERS}")
message(STATUS "  Source files:      ${K3_ALL_SRC}")
message(STATUS "")
message(STATUS "Next steps:")
message(STATUS "  cmake --build ${CMAKE_BINARY_DIR} -j")
message(STATUS "  ctest --test-dir ${CMAKE_BINARY_DIR} --output-on-failure")
message(STATUS "  cmake --install ${CMAKE_BINARY_DIR}")