# Provides
#   See https://cmake.org/cmake/help/latest/module/FindCUDA.html
if (${MNM_USE_CUDA} STREQUAL "OFF")
  message(STATUS "Build without CUDA")
else()
  find_package(CUDA REQUIRED)
  message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
endif()
