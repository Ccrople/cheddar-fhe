# Copyright 2026
# Licensed under the Apache License, Version 2.0.
# SPDX-License-Identifier: Apache-2.0
#
# The plaintext oracle builds STANDALONE and is not referenced by the root
# CMakeLists.txt. That is deliberate:
#
#   * the root CMakeLists.txt is a shared integration file owned by S0 during
#     wave 1, so S2 does not edit it;
#   * the oracle is pure C++17 with no CUDA, no RMM, no googletest and no
#     network access at configure time, while the Cheddar build requires all
#     of those -- keeping it separate means the reference can be built and run
#     on a laptop with no GPU;
#   * `add_subdirectory(oracle)` is a one-line change S0 can make whenever the
#     encrypted tests need to link against it. The patch is in
#     oracle/docs/ORACLE_HANDOFF.md.
#
# Build:
#   cmake -S oracle -B oracle/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build oracle/build --config Release
#   ctest --test-dir oracle/build -C Release --output-on-failure

cmake_minimum_required(VERSION 3.16)
project(cheddar_llama3_oracle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Floating-point determinism. The oracle's whole value is that its answer does
# not move, so contraction and fast-math are switched off explicitly rather
# than left to the compiler's default for the build type.
if(MSVC)
  add_compile_options(/fp:precise /W4)
else()
  add_compile_options(-ffp-contract=off -fno-fast-math -Wall -Wextra)
endif()

add_library(llama3_oracle STATIC
  src/Calibrate.cpp
  src/Cheby.cpp
  src/Compare.cpp
  src/Config.cpp
  src/Json.cpp
  src/Ops.cpp
  src/Serialize.cpp
  src/Tensor.cpp
  src/Vectors.cpp
  src/Weights.cpp
)
target_include_directories(llama3_oracle PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(llama3_oracle_cli tools/oracle_main.cpp)
target_link_libraries(llama3_oracle_cli PRIVATE llama3_oracle)

enable_testing()
add_executable(llama3_oracle_test tests/Check.cpp tests/OracleTest.cpp)
target_link_libraries(llama3_oracle_test PRIVATE llama3_oracle)
add_test(NAME oracle_unit_tests COMMAND llama3_oracle_test)

# A second gate: the CLI must run its own vector self-test cleanly. This
# catches a report path that compiles but throws, which the unit tests do not
# exercise.
add_test(NAME oracle_cli_selftest
         COMMAND llama3_oracle_cli selftest --reduced --tokens=8)
add_test(NAME oracle_cli_config COMMAND llama3_oracle_cli config)
add_test(NAME oracle_cli_boundaries COMMAND llama3_oracle_cli boundaries)
add_test(NAME oracle_cli_thresholds COMMAND llama3_oracle_cli thresholds)
add_test(NAME oracle_cli_calibrate
         COMMAND llama3_oracle_cli calibrate --reduced --tokens=16 --layers=2)
