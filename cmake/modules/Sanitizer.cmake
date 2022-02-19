# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Provides:
#   - MNM_CXX_SANITIZER_FLAGS
#   - MNM_CUDA_SANITIZER_FLAGS
#   - mnm_target_add_sanitizer: macro, mutate the target
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

macro(_mnm_sanitizer_flags lang sanitizer)
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

macro(mnm_target_add_sanitizer target)
  if (NOT ${MNM_USE_SANITIZER} STREQUAL "OFF")
    set_property(TARGET ${target} APPEND PROPERTY
      COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:SHELL:${MNM_CUDA_SANITIZER_FLAGS}>)
    set_property(TARGET ${target} APPEND PROPERTY
      COMPILE_OPTIONS $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:SHELL:${MNM_CXX_SANITIZER_FLAGS}>)
    target_link_options(${target} PUBLIC $<$<BOOL:TRUE>:SHELL:${MNM_CXX_SANITIZER_FLAGS}>)
  endif()
endmacro()

set(MNM_CXX_SANITIZER_FLAGS "")
set(MNM_CUDA_SANITIZER_FLAGS "")
if (${MNM_USE_SANITIZER} STREQUAL "OFF")
  message (STATUS "Build without sanitizer")
elseif (${MNM_USE_SANITIZER} IN_LIST _SANITIZERS)
  _mnm_sanitizer_flags(CXX ${MNM_USE_SANITIZER})
  set(MNM_CXX_SANITIZER_FLAGS ${CXX_${MNM_USE_SANITIZER}_FLAGS})
  message(STATUS "Found MNM_CXX_SANITIZER_FLAGS = ${MNM_CXX_SANITIZER_FLAGS}")
  if (NOT ${MNM_USE_CUDA} STREQUAL "OFF")
    _mnm_sanitizer_flags(CUDA ${MNM_USE_SANITIZER})
    set(MNM_CUDA_SANITIZER_FLAGS ${CUDA_${MNM_USE_SANITIZER}_FLAGS})
    message(STATUS "Found MNM_CUDA_SANITIZER_FLAGS = ${MNM_CUDA_SANITIZER_FLAGS}")
  endif()
else ()
  message(FATAL_ERROR "Cannot recognize sanitizer: ${MNM_USE_SANITIZER}")
endif ()
