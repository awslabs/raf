cmake_minimum_required(VERSION 3.8)
project(include_mnm C CXX)

add_executable(include_mnm main.cc)
target_include_directories(include_mnm PRIVATE ${MNM_INCLUDE_DIRS})
target_link_libraries(include_mnm PRIVATE mnm ${MNM_LINK_LIBS})
target_compile_features(include_mnm PRIVATE cxx_std_14)

set_target_properties(include_mnm PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)
