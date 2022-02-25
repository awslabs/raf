# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Provides
#   - GIT_FOUND - true if the command line client was found
#   - GIT_EXECUTABLE - path to git command line client
#   - GIT_VERSION_STRING - the version of git found (since CMake 2.8.8)
#   - RAF_GIT_VERSION
find_package(Git QUIET)
if (${GIT_FOUND})
  message(STATUS "Git found: ${GIT_EXECUTABLE}")
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                  OUTPUT_VARIABLE RAF_GIT_VERSION
                  ERROR_QUIET
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
  message(WARNING "Git not found")
  set(RAF_GIT_VERSION "git-not-found")
endif()
