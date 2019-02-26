macro(clangformat)

string(REPLACE ":" ";" _PATH $ENV{PATH})
foreach(p ${_PATH})
  file(GLOB cand ${p}/clang-format*)
  if(cand)
    set(CLANG_FORMAT_EXECUTABLE ${cand})
    set(CLANG_FORMAT_FOUND ON)
    execute_process(COMMAND ${CLANG_FORMAT_EXECUTABLE} -version OUTPUT_VARIABLE clang_out )
    string(REGEX MATCH .*\(version[^\n]*\)\n version ${clang_out})
    set(CLANG_FORMAT_VERSION ${CMAKE_MATCH_1})
    break()
  else()
    set(CLANG_FORMAT_FOUND OFF)
  endif()
endforeach()

if (CLANG_FORMAT_FOUND)
  execute_process(COMMAND git rev-parse --git-dir OUTPUT_VARIABLE GIT_DIR)
  string(STRIP ${GIT_DIR} GIT_DIR)
  execute_process(COMMAND git rev-parse --show-toplevel OUTPUT_VARIABLE REPO_DIR)
  string(STRIP ${REPO_DIR} REPO_DIR)
  if (NOT EXISTS ${GIT_DIR}/hooks/pre-commit)
    message(STATUS "Enable pre-commit code linting.")
    execute_process(COMMAND ln -s ${CMAKE_SOURCE_DIR}/scripts/pre-commit)
  else()
    execute_process(
      COMMAND find ${GIT_DIR}/hooks -type l -ls
      COMMAND grep "/scripts/pre-commit"
      COMMAND awk "{ print $NF }"
      COMMAND tr -d "\n"
      OUTPUT_VARIABLE PRECOMMIT_LINKED)
    if (${PRECOMMIT_LINKED} STREQUAL ${REPO_DIR}/scripts/pre-commit)
      message(STATUS "Already pre-commit hook linked!")
    else()
      message(WARNING "Something else has already been hooked in pre-commit!")
    endif()
  endif()
endif()

endmacro()
