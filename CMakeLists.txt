cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# CUDA 13.0 removed offline compilation below compute capability 7.5, so the
# default list cannot carry 60/61/70 any more. This has to sit *before*
# project(): enable_language(CUDA) fills CMAKE_CUDA_ARCHITECTURES in with the
# compiler's own default (a single architecture), after which a guard here can
# never fire. It has to be a guard rather than the unconditional set() that was
# here before, because that one shadowed -DCMAKE_CUDA_ARCHITECTURES on the
# command line and silently built every architecture instead of the one asked
# for. Blackwell wants -DCMAKE_CUDA_ARCHITECTURES=100 and a toolkit >= 12.8.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set (CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
endif()

project(cheddar LANGUAGES CUDA CXX)

# Using C++17 standard
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CUDA_STANDARD 17)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Automatically detect the CUDA architecture
#if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#  message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined, setting to native")
#  set (CMAKE_CUDA_ARCHITECTURES native CACHE STRING "Auto-detect CUDA architecture")
#endif()

# dependencies
find_package(CUDAToolkit 11.8 REQUIRED)

include(FetchContent)
option(BUILD_TESTS OFF)
FetchContent_Declare(
  rmm
  GIT_REPOSITORY https://github.com/rapidsai/rmm
  GIT_TAG        branch-22.12
  GIT_SHALLOW
)
FetchContent_MakeAvailable(rmm)
message(STATUS "RMM source dir: ${rmm_SOURCE_DIR}")

# options
option(ENABLE_EXTENSION "Enable extension sources" ON)
option(BUILD_UNITTEST "Build unit tests" ON)
option(USE_GMP "Use GMP instead of libtommath" OFF)
option(USE_CUBLAS "Use cuBLAS for the plaintext matrix products in PCMM" OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# source files
set(CKKS_GPU_SOURCES
  # src/core/BigInt.cpp
  src/core/Container.cpp
  src/core/Context.cpp
  src/core/DeviceVector.cpp
  src/core/Streams.cpp
  src/core/Encode.cpp
  src/core/EncodeGpu.cu
  src/core/EvkMap.cpp
  src/core/EvkRequest.cpp
  src/core/MemoryPool.cpp
  src/core/MultiLevelCiphertext.cpp
  src/core/NPInfo.cpp
  src/core/Serialization.cpp
  src/core/ElementWise.cu
  src/core/ModSwitch.cu
  src/core/NTT.cu
  src/core/Mlwe.cu
  src/core/SubringMatrix.cu
  src/core/SubringCtMatrix.cu
  src/core/BatchCcmm.cu
  src/core/Cmt.cu
  src/core/Parameter.cu
  src/core/Pcmm.cu
  src/core/PcmmBlas.cu
  src/core/CiLift.cu
  src/core/CiSwitchedCcmm.cu
  src/core/RingSwitch.cu
  src/core/SwitchedCcmm.cu
  src/core/CtAxisExchange.cu
  src/UserInterface.cu
)
if(USE_GMP)
  list(APPEND CKKS_GPU_SOURCES src/core/BigInt_gmp.cpp)
else()
  list(APPEND CKKS_GPU_SOURCES src/core/BigInt_tommath.cpp)
endif()

if(ENABLE_EXTENSION)
  add_compile_definitions(ENABLE_EXTENSION)
  list(APPEND CKKS_GPU_SOURCES
    src/extension/AttentionPacking.cpp
    src/extension/BootContext.cpp
    src/extension/CiSinCAttention.cu
    src/extension/CiLlamaSeam.cu
    src/extension/CiLlamaLayer.cu
    src/extension/CiLayerPrefetch.cu
    src/extension/CiBatch.cu
    src/extension/CiBatchLayer.cu
    src/extension/CiModuleBasis.cpp
    src/extension/CiSinCBasis.cpp
    src/extension/BootParameter.cpp
    src/extension/ComplexLinearTransform.cpp
    src/extension/EvalMod.cpp
    src/extension/EvalPoly.cpp
    src/extension/EvalSpecialFFT.cpp
    src/extension/Hoist.cu
    src/extension/LinearTransform.cpp
    src/extension/LlamaAttention.cu
    src/extension/LlamaBlock.cu
    src/extension/LlamaLinear.cu
    src/extension/Profile.cpp
  src/extension/SlotPermute.cpp
    src/extension/SinCAttention.cu
    src/extension/RmsNorm.cu
    src/extension/SiLu.cu
    src/extension/SoftMax.cu
    src/extension/RoPe.cu
    src/extension/StripedMatrix.cpp
    src/extension/SylphSchedule.cpp
  )
endif()

message(STATUS "CKKS_GPU_SOURCES: ${CKKS_GPU_SOURCES}")

# build library
add_library(cheddar SHARED ${CKKS_GPU_SOURCES})

if(USE_CUBLAS)
  # cuBLAS is not part of the CUDA toolkit prefix on every machine here, so the
  # location is configurable rather than assumed. [SYLPH] section 2.2 credits
  # its linear-algebra speed to BLAS offload, and the measured alternative on
  # this card is 3-4x slower.
  find_library(CUBLAS_LIB cublas
               HINTS ${CUBLAS_ROOT}/lib ${CUDAToolkit_LIBRARY_DIR} REQUIRED)
  target_compile_definitions(cheddar PUBLIC USE_CUBLAS)
  target_link_libraries(cheddar PUBLIC ${CUBLAS_LIB})
  if(CUBLAS_ROOT)
    target_include_directories(cheddar PUBLIC ${CUBLAS_ROOT}/include)
  endif()
endif()

if(USE_GMP)
  find_library(GMP gmp REQUIRED)
  set(MATH_LIB gmp)
  target_compile_definitions(cheddar PUBLIC USE_GMP)
else()
  find_library(LIBTOMMATH tommath libtommath REQUIRED)
  set(MATH_LIB ${LIBTOMMATH})
endif()

# The host-side matrix composition behind the compiled transforms runs on
# every core (common/ParallelFor.h).
find_package(Threads REQUIRED)
target_link_libraries(cheddar
  PUBLIC CUDA::cudart rmm ${MATH_LIB} Threads::Threads
)

target_include_directories(cheddar
  PUBLIC include
)

target_compile_options(cheddar PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-O0>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELWITHDEBINFO>>:-lineinfo>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-O3>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:DEBUG>>:-O0>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RELWITHDEBINFO>>:-lineinfo>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RELEASE>>:-O3>
)

# subdirectories
if (BUILD_UNITTEST)
  message(STATUS "Building unit tests")
  add_subdirectory(unittest)
endif()
