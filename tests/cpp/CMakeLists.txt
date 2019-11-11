# Compile test_*.cc under the current folder
file(GLOB_RECURSE MNM_CPPTEST_SRCS ${CMAKE_CURRENT_LIST_DIR}/test_*.cc)
# All executable targets
set(MNM_CPPTEST_TARGETS "")

# Create all the targets
foreach(__src ${MNM_CPPTEST_SRCS})
  # Figure out the name of test targets
  get_filename_component(__filename ${__src} NAME)
  string(FIND ${__filename} ".cc" __pos REVERSE)
  string(REPLACE ".cc" "" __target ${__filename})
  # Targets are named as mnm_cpptest_test_*.cc
  set(__target "mnm_cpptest_${__target}")
  list(APPEND MNM_CPPTEST_TARGETS ${__target})
  # Add test
  mnm_test(${__target} ${__src})
endforeach()

# Create a target that includes all testcases
add_custom_target(mnm-cpptest DEPENDS ${MNM_CPPTEST_TARGETS})
unset(MNM_CPPTEST_TARGETS)
unset(MNM_CPPTEST_SRCS)
