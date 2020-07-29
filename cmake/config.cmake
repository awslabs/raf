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

########## MNM Configuration ##########
# Convention: MNM_USE_LIB could be ON/OFF or a string indicating path to LIB

# MNM_USE_LLVM. Option: [ON/OFF/Path-to-llvm-config-executable]"
set(MNM_USE_LLVM ON)

# MNM_USE_GTEST. Option: [ON/OFF]
set(MNM_USE_GTEST ON)

# MNM_USE_CUDA. Option: [ON/OFF]
set(MNM_USE_CUDA OFF)

# MNM_USE_CUBLAS. Option: [ON/OFF]
set(MNM_USE_CUBLAS OFF)

# MNM_USE_CUDNN. Option: [ON/OFF/Path-To-CUDNN]. You may use environment variables, like $ENV{CUDNN_HOME}
set(MNM_USE_CUDNN OFF)

# MNM_USE_SANITIZER. Option: [OFF/ASAN/MSAN/TSAN/UBSAN]"
set(MNM_USE_SANITIZER OFF)
