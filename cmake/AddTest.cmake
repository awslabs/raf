function(add_test SRCS INC_DIRS LINK_LIBS TARGET IS_CUDA)

set(EXECS "")

foreach(__src ${SRCS})
  get_filename_component(__name ${__src} NAME)
  string(FIND ${__name} ".cc" __pos REVERSE)

  if (__pos EQUAL -1)
    string(REPLACE ".cu" "" __exec ${__name})
  else()
    string(REPLACE ".cc" "" __exec ${__name})
  endif()

  #if(${IS_CUDA})
    #cuda_add_executable(${__exec} ${__src})
  #else()
    add_executable(${__exec} ${__src})
  #endif()

  list(APPEND EXECS ${__exec})
  # TODO(@were): try something that could compile on Windows
  foreach (__lib ${LINK_LIBS})
    target_link_libraries(${__exec} ${__lib})
  endforeach()

  foreach (__dirs ${INC_DIRS})
    string(REPLACE " " ";" dirs ${__dirs})
    foreach (__dir ${dirs})
      target_include_directories(${__exec} PRIVATE ${__dir})
    endforeach()
  endforeach()

  target_compile_features(${__exec} PRIVATE cxx_std_14)
  set_target_properties(${__exec} PROPERTIES EXCLUDE_FROM_ALL 1)
  set_target_properties(${__exec} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
  set_target_properties(${__exec} PROPERTIES CXX_STANDARD 14)
endforeach()

add_custom_target(${TARGET} DEPENDS ${EXECS})

endfunction()
aaaa
