# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Provides:
#   - RAF_CXX_SANITIZER_FLAGS
#   - RAF_CUDA_SANITIZER_FLAGS
#   - raf_target_add_sanitizer: macro, mutate the target
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
include(CheckFortranCompilerFlag)
include(${PROJECT_SOURCE_DIR}/cmake/utils/CheckCUDACompilerFlag.cmake)

set(_SANITIZERS "ASAN" "MSAN" "TSAN" "UBSAN")

set(_ASAN_POSSIBLE_FLAGS
  "-g -fsanitize=address -fno-omit-frame-pointer"
  "-g -fsanitize=address"
  "-g -faddress-sanitizer"
  "-g -Xcompiler -g -Xcompiler -fsanitize=address -Xcompiler -fno-omit-frame-pointer"
  "-g -Xcompiler -g -Xcompiler -fsanitize=address"
  "-g -Xcompiler -g -Xcompiler -faddress-sanitizer"
)

set(_MSAN_POSSIBLE_FLAGS
  "-g -fsanitize=memory"
  "-g -Xcompiler -g -Xcompiler -fsanitize=memory"
)

set(_TSAN_POSSIBLE_FLAGS
  "-g -fsanitize=thread"
  "-g -Xcompiler -g -Xcompiler -fsanitize=thread"
)

set(_UBSAN_POSSIBLE_FLAGS
  "-g -fsanitize=undefined"
  "-g -Xcompiler -g -Xcompiler -fsanitize=undefined"
)

macro(_raf_sanitizer_flags lang sanitizer)
  # Provides:
  #   - ${lang}_SUPPORT_${sanitizer}
  #   - ${lang}_{sanitizer}_FLAGS
  set(${lang}_${sanitizer}_FLAGS "")
  set(__flags ${_${sanitizer}_POSSIBLE_FLAGS})
  set(__compiler ${CMAKE_${lang}_COMPILER_ID})
  foreach (__flag ${__flags})
    set(CMAKE_REQUIRED_FLAGS "${__flag}")
    if (${lang} STREQUAL "C")
      check_c_compiler_flag("${__flag}" C_SUPPORT_${sanitizer})
    elseif (${lang} STREQUAL "CXX")
      check_cxx_compiler_flag("${__flag}" CXX_SUPPORT_${sanitizer})
    elseif (${lang} STREQUAL "FORTRAN")
      check_fortran_compiler_flag("${__flag}" FORTRAN_SUPPORT_${sanitizer})
    elseif (${lang} STREQUAL "CUDA")
      check_cuda_compiler_flag("${__flag}" CUDA_SUPPORT_${sanitizer})
    else()
      message(WARNING "Unknown language ${lang}")
    endif()
    if (${lang}_SUPPORT_${sanitizer})
      set(${lang}_${sanitizer}_FLAGS "${__flag}")
      message(STATUS "Found ${lang}_${sanitizer}_FLAGS = ${__flag}")
      break()
    endif ()
    unset(${lang}_SUPPORT_${sanitizer} CACHE)
  endforeach ()
  if (NOT ${lang}_SUPPORT_${sanitizer})
    message(STATUS "Not found ${lang}_${sanitizer}_FLAGS")
  endif ()
endmacro()

macro(raf_target_add_sanitizer target)
  if (NOT ${RAF_USE_SANITIZER} STREQUAL "OFF")
    set_property(TARGET ${target} APPEND PROPERTY
      COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:SHELL:${RAF_CUDA_SANITIZER_FLAGS}>)
    set_property(TARGET ${target} APPEND PROPERTY
      COMPILE_OPTIONS $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:SHELL:${RAF_CXX_SANITIZER_FLAGS}>)
    target_link_options(${target} PUBLIC $<$<BOOL:TRUE>:SHELL:${RAF_CXX_SANITIZER_FLAGS}>)
  endif()
endmacro()

set(RAF_CXX_SANITIZER_FLAGS "")
set(RAF_CUDA_SANITIZER_FLAGS "")
if (${RAF_USE_SANITIZER} STREQUAL "OFF")
  message (STATUS "Build without sanitizer")
elseif (${RAF_USE_SANITIZER} IN_LIST _SANITIZERS)
  _raf_sanitizer_flags(CXX ${RAF_USE_SANITIZER})
  set(RAF_CXX_SANITIZER_FLAGS ${CXX_${RAF_USE_SANITIZER}_FLAGS})
  message(STATUS "Found RAF_CXX_SANITIZER_FLAGS = ${RAF_CXX_SANITIZER_FLAGS}")
  if (NOT ${RAF_USE_CUDA} STREQUAL "OFF")
    _raf_sanitizer_flags(CUDA ${RAF_USE_SANITIZER})
    set(RAF_CUDA_SANITIZER_FLAGS ${CUDA_${RAF_USE_SANITIZER}_FLAGS})
    message(STATUS "Found RAF_CUDA_SANITIZER_FLAGS = ${RAF_CUDA_SANITIZER_FLAGS}")
  endif()
else ()
  message(FATAL_ERROR "Cannot recognize sanitizer: ${RAF_USE_SANITIZER}")
endif ()
