##############################################################################
# Provide:
#  - MNM_NCCL_LIBRARY
#  - MNM_NCCL_INCLUDE

if (${MNM_USE_NCCL} STREQUAL "OFF")
  message(STATUS "Build without NCCL support")
  set(MNM_NCCL_INCLUDE "")
  set(MNM_NCCL_LIBRARY "")
else()
  if (${MNM_USE_CUDA} STREQUAL "OFF")
    message(FATAL_ERROR "Cannot enable NCCL without using CUDA.")
  endif()
  if (${MNM_USE_MPI} STREQUAL "OFF")
    message(FATAL_ERROR "Cannot enable NCCL without using MPI now.")
  endif()
  find_package(NCCL REQUIRED)
  set(MNM_NCCL_INCLUDE ${NCCL_INCLUDE_DIRS})
  message(STATUS "Found MNM_NCCL_INCLUDE = ${MNM_NCCL_INCLUDE}")
  set(MNM_NCCL_LIBRARY ${NCCL_LIBRARIES})
  message(STATUS "Found MNM_NCCL_LIBRARY = ${MNM_NCCL_LIBRARY}")
endif()