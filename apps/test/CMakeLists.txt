include_directories(${MNM_HEADER_FILES})

add_executable(test main.cc)
target_link_libraries(test mnm)
target_compile_features(test PRIVATE cxx_std_17)
target_include_directories(mnm PRIVATE ${MNM_HEADER_FILES})
