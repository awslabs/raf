include(${CMAKE_SOURCE_DIR}/cmake/FindGTEST.cmake)

find_gtest(${MNM_USE_GTEST})

if (NOT GTEST_FOUND)
  message(STATUS "No gtest found! mnm-cpptest target disabled!")
  message(STATUS "You can enable it by setting the USE_GTEST variable, and re-cmake it!")
  return()
endif()

file(GLOB_RECURSE SRCS ./*.cc)

foreach(__src ${SRCS})
  get_filename_component(__name ${__src} NAME)
  string(FIND ${__name} ".cc" __pos REVERSE)

  if (__pos EQUAL -1)
    string(REPLACE ".cu" "" __exec ${__name})
  else()
    string(REPLACE ".cc" "" __exec ${__name})
  endif()

  add_executable(${__exec} ${__src})

  list(APPEND EXECS ${__exec})
  # TODO(@were): try something that could compile on Windows
  target_link_libraries(${__exec} mnm)
  target_link_libraries(${__exec} ${CMAKE_DL_LIBS})
  target_link_libraries(${__exec} pthread)
  target_link_libraries(${__exec} ${GTEST_LIBRARY})

  target_include_directories(${__exec} PRIVATE ${MNM_HEADER_FILES})
  target_include_directories(${__exec} PRIVATE ${MNM_3RD_PARTY_INCLUDE_DIRS})
  target_include_directories(${__exec} PRIVATE ${GTEST_INCLUDE_DIR})

  target_compile_features(${__exec} PRIVATE cxx_std_14)
  set_target_properties(${__exec} PROPERTIES EXCLUDE_FROM_ALL 1)
  set_target_properties(${__exec} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
  set_target_properties(${__exec} PROPERTIES CXX_STANDARD 14)
endforeach()

add_custom_target(mnm-cpptest DEPENDS ${EXECS})
