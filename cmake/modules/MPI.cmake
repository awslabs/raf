# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

##############################################################################
# Provide:
#  - RAF_MPI_LIBRARY
#  - RAF_MPI_INCLUDE

if (${RAF_USE_MPI} STREQUAL "OFF")
  message(STATUS "Build without MPI support")
  set(RAF_MPI_INCLUDE "")
  set(RAF_MPI_LIBRARY "")
else()
  find_package(MPI REQUIRED)
  add_definitions(-DOMPI_SKIP_MPICXX)
  set(RAF_MPI_INCLUDE ${MPI_INCLUDE_PATH})
  message(STATUS "Found RAF_MPI_INCLUDE = ${RAF_MPI_INCLUDE}")
  set(RAF_MPI_LIBRARY ${MPI_CXX_LIBRARIES})
  message(STATUS "Found RAF_MPI_LIBRARY = ${RAF_MPI_LIBRARY}")
endif()