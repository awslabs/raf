macro(find_gtest use_gtest)


########################################################
# Find the GTEST library
#
# Variables Provide
# -----------------
# GTEST_BOTH_LIBRARIES - Both libgtest & libgtest-main
# GTEST_LIBRARIES - libgtest
# GTEST_MAIN_LIBRARIES - libgtest-main
#

if (IS_DIRECTORY ${use_gtest})
  # Provide the prefix of gtest installed
  set(GTEST_ROOT ${use_gtest})
else()
  if (NOT ${use_gtest})
    message("GTest is not enabled! Skip!")
    return()
  endif()
endif()

find_package(GTest REQUIRED)

endmacro()
