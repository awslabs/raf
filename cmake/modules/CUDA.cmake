# Provides:
#  - MNM_CUDA_INCLUDE
#
#  See https://cmake.org/cmake/help/latest/module/FindCUDA.html
if (${MNM_USE_CUDA} STREQUAL "OFF")
  message(STATUS "Build without CUDA")
  set(MNM_CUDA_INCLUDE "")
else()
  find_package(CUDA REQUIRED)
  message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
  set(MNM_CUDA_INCLUDE ${CUDA_INCLUDE_DIRS})
  message(STATUS "MNM_CUDA_INCLUDE = ${MNM_CUDA_INCLUDE}")
endif()
