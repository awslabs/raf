##############################################################################
# Provide:
#  - MNM_MPI_LIBRARY
#  - MNM_MPI_INCLUDE

if (${MNM_USE_MPI} STREQUAL "OFF")
  message(STATUS "Build without MPI support")
  set(MNM_MPI_INCLUDE "")
  set(MNM_MPI_LIBRARY "")
else()
  find_package(MPI REQUIRED)
  add_definitions(-DOMPI_SKIP_MPICXX)
  set(MNM_MPI_INCLUDE ${MPI_INCLUDE_PATH})
  message(STATUS "Found MNM_MPI_INCLUDE = ${MNM_MPI_INCLUDE}")
  set(MNM_MPI_LIBRARY ${MPI_CXX_LIBRARIES})
  message(STATUS "Found MNM_MPI_LIBRARY = ${MNM_MPI_LIBRARY}")
endif()