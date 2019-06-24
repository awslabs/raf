macro(find_cuda_and_cudnn use_cuda use_cudnn)

#######################################################
# MNM-specialized version of find CUDA.
#
# Usage:
#   find_cuda_and_cudnn(${MNM_USE_CUDA} ${MNM_USE_CUDNN})
#
# Provide variables:
#
# - CUDA_FOUND
# - CUDA_INCLUDE_DIRS
# - CUDA_TOOLKIT_ROOT_DIR
# - CUDA_CUDA_LIBRARY
# - CUDA_CUDART_LIBRARY
# - CUDA_NVRTC_LIBRARY
# - CUDA_CUDNN_LIBRARY
# - CUDA_CUBLAS_LIBRARY
#

set(CUDA_CUDNN_LIBRARY "CUDA_CUDNN_LIBRARY-NOTFOUND")

include(${CMAKE_SOURCE_DIR}/3rdparty/tvm/cmake/util/FindCUDA.cmake)
find_cuda(${use_cuda})

if (NOT CUDA_CUDNN_LIBRARY)
  if (IS_DIRECTORY ${use_cudnn})
    find_library(CUDA_CUDNN_LIBRARY cudnn
                 ${use_cudnn}/lib
                 ${use_cudnn}/lib64)
    if (NOT CUDA_CUDNN_LIBRARY)
      message(FATAL_ERROR "Cannot find CUDNN with the given hint!")
    else()
      message(STATUS "Found in another path: CUDA_CUDNN_LIBRARY=${CUDA_CUDNN_LIBRARY}")
      set(CUDNN_INCLUDE_DIR "${use_cudnn}/include")
      message(STATUS "Set CUDNN_INCLUDE_DIR=${CUDNN_INCLUDE_DIR}")
    endif()
  else()
    if (${use_cudnn})
      find_library(CUDA_CUDNN_LIBRARY cudnn
                   $ENV{CUDNN_HOME}/lib
                   $ENV{CUDNN_HOME}/lib64)
      set(CUDNN_INCLUDE_DIR "$ENV{CUDNN_HOME}/include")
    else()
      set(CUDNN_INCLUDE_DIR OFF)
    endif()
  endif()
endif()

endmacro()
