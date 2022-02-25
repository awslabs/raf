# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Compile test_*.cc under the current folder
file(GLOB_RECURSE RAF_CPPTEST_SRCS ${CMAKE_CURRENT_LIST_DIR}/test_*.cc)
# All executable targets
set(RAF_CPPTEST_TARGETS "")

# Create all the targets
foreach(__src ${RAF_CPPTEST_SRCS})
  # Figure out the name of test targets
  get_filename_component(__filename ${__src} NAME)
  string(FIND ${__filename} ".cc" __pos REVERSE)
  string(REPLACE ".cc" "" __target ${__filename})
  # Targets are named as raf_cpptest_test_*.cc
  set(__target "raf_cpptest_${__target}")
  list(APPEND RAF_CPPTEST_TARGETS ${__target})
  # Add test
  raf_test(${__target} ${__src})
endforeach()

# Create a target that includes all testcases
add_custom_target(raf-cpptest DEPENDS ${RAF_CPPTEST_TARGETS})
unset(RAF_CPPTEST_TARGETS)
unset(RAF_CPPTEST_SRCS)
