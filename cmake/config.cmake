# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# The example config.cmake
# When building the project, copy this file to the `build' directory

########## CMake Configuration #########
# Below are the suggested configurations to CMake
set(BUILD_SHARED_LIBS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
set(CMAKE_C_COMPILER_LAUNCHER ccache)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

########## MNM Configuration ##########
# Convention: MNM_USE_LIB could be ON/OFF or a string indicating path to LIB

# MNM_USE_LLVM. Option: [Path-to-llvm-config-executable]"
# Note 1: We do not support "ON" because we enforce LLVM to be linked
#         statically so that we could hide its symbols to work with PyTorch 1.10+.
# Note 2: You may need to change llvm-config-8 according to your environment.
set(MNM_USE_LLVM llvm-config-8)
set(HIDE_PRIVATE_SYMBOLS ON)

# MNM_USE_GTEST. Option: [ON/OFF]
set(MNM_USE_GTEST ON)

# MNM_USE_CUDA. Option: [ON/OFF]
set(MNM_USE_CUDA OFF CACHE STRING "Build with CUDA" FORCE)

# CUDA architecture. Option: 70(V100), 75(T4), 80(A100)
set(MNM_CUDA_ARCH 70 CACHE STRING "CUDA architure" FORCE)

# MNM_USE_CUBLAS. Option: [ON/OFF]
set(MNM_USE_CUBLAS OFF CACHE STRING "Build with CuBLAS" FORCE)

# MNM_USE_CUDNN. Option: [ON/OFF/Path-To-CUDNN]. You may use environment variables, like $ENV{CUDNN_HOME}
set(MNM_USE_CUDNN OFF CACHE STRING "Build with CuDNN" FORCE)

# MNM_USE_SANITIZER. Option: [OFF/ASAN/MSAN/TSAN/UBSAN]"
set(MNM_USE_SANITIZER OFF)

# MNM_USE_MPI. Option: [ON/OFF]
set(MNM_USE_MPI OFF CACHE STRING "Build with MPI" FORCE)

# MNM_USE_NCCL. Option: [ON/OFF]
set(MNM_USE_NCCL OFF CACHE STRING "Build with NCCL" FORCE)

# MNM_USE_CUTLASS. Option: [ON/OFF].
set(MNM_USE_CUTLASS OFF CACHE STRING "Build with CUTLASS" FORCE)
