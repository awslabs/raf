cmake_minimum_required(VERSION 3.8)
project(test_mnm C CXX)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)

add_executable(test_mnm main.cc)
target_compile_features(test_mnm PRIVATE cxx_std_14)
target_include_directories(test_mnm PRIVATE ${MNM_HEADER_FILES})
target_include_directories(test_mnm PRIVATE ${MNM_3RD_PARTY_INCLUDE_DIRS})
target_link_libraries(test_mnm mnm)
target_link_libraries(test_mnm dl)
