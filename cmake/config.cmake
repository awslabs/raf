# The example config.cmake
# When building the project, copy this file to the `build' directory

########## CMake Configuration #########
# Below are the suggested configurations to CMake
set(BUILD_SHARED_LIBS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type")
set(CMAKE_C_COMPILER_LAUNCHER ccache)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
set(CMAKE_CUDA_COMPILER_LAUNCHER ccache)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

########## MNM Configuration ##########
# Convention: MNM_USE_LIB could be ON/OFF or a string indicating path to LIB

# MNM_USE_LLVM. Option: [ON/OFF/Path-to-llvm-config-executable]"
set(MNM_USE_LLVM ON)

# MNM_USE_GTEST. Option: [ON/OFF]
set(MNM_USE_GTEST ON)

# MNM_USE_CUDA. Option: [ON/OFF]
set(MNM_USE_CUDA OFF)

# CUDA architecture. Option: 70(V100), 75(T4), 80(A100)
set(MNM_CUDA_ARCH 70)

# MNM_USE_CUBLAS. Option: [ON/OFF]
set(MNM_USE_CUBLAS OFF)

# MNM_USE_CUDNN. Option: [ON/OFF/Path-To-CUDNN]. You may use environment variables, like $ENV{CUDNN_HOME}
set(MNM_USE_CUDNN OFF)

# MNM_USE_SANITIZER. Option: [OFF/ASAN/MSAN/TSAN/UBSAN]"
set(MNM_USE_SANITIZER OFF)

# MNM_USE_MPI. Option: [ON/OFF]
set(MNM_USE_MPI OFF)

# MNM_USE_NCCL. Option: [ON/OFF]
set(MNM_USE_NCCL OFF)

# MNM_USE_CUTLASS. Option: [ON/OFF].
set(MNM_USE_CUTLASS OFF)
