# The example config.cmake
# When building the probject, copy this file to the `build' directory

########################################################################################
# MNM variables. In avoid of poluting TVM's build configuration, all the MNM-dedicated
# config-variables start with "MNM" prefix.
#

# USE_CUDA can their to "ON", the path to CUDA.
set(MNM_USE_CUDA OFF)
set(MNM_USE_CUDNN OFF)

# We do not submodule gtest, so the c++/cuda test module is optional.
# If the test module is desired, set it to either "ON" or the prefix to "GTEST"!
set(MNM_USE_GTEST OFF)
