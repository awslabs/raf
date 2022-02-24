# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# The example config.cmake
# When building the project, copy this file to the `build' directory

########## CMake Configuration #########
# Below are the suggested configurations to CMake
set(BUILD_SHARED_LIBS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_COMPILER_LAUNCHER ccache)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

########## RAF Configuration ##########
# Convention: RAF_USE_LIB could be ON/OFF or a string indicating path to LIB

# RAF_USE_LLVM. Option: [Path-to-llvm-config-executable]"
# Note 1: We do not support "ON" because we enforce LLVM to be linked
#         statically so that we could hide its symbols to work with PyTorch 1.10+.
# Note 2: You may need to change llvm-config-8 according to your environment.
set(RAF_USE_LLVM llvm-config-8)
set(HIDE_PRIVATE_SYMBOLS ON)

# RAF_USE_GTEST. Option: [ON/OFF]
set(RAF_USE_GTEST ON)

# RAF_USE_CUDA. Option: [ON/OFF]
set(RAF_USE_CUDA OFF)

# CUDA architecture. Option: 70(V100), 75(T4), 80(A100)
set(RAF_CUDA_ARCH 70)

# RAF_USE_CUBLAS. Option: [ON/OFF]
set(RAF_USE_CUBLAS OFF)

# RAF_USE_CUDNN. Option: [ON/OFF/Path-To-CUDNN]. You may use environment variables, like $ENV{CUDNN_HOME}
set(RAF_USE_CUDNN OFF)

# RAF_USE_SANITIZER. Option: [OFF/ASAN/MSAN/TSAN/UBSAN]"
set(RAF_USE_SANITIZER OFF)

# RAF_USE_MPI. Option: [ON/OFF]
set(RAF_USE_MPI OFF)

# RAF_USE_NCCL. Option: [ON/OFF]
set(RAF_USE_NCCL OFF)

# RAF_USE_CUTLASS. Option: [ON/OFF].
set(RAF_USE_CUTLASS OFF)
