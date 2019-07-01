if (MNM_USE_CUDNN)

  message(STATUS "Enabling CUDNN backend...")

  include(${CMAKE_SOURCE_DIR}/cmake/FindCUDA.cmake)
  find_cuda_and_cudnn(${MNM_USE_CUDA} ${MNM_USE_CUDNN})

  # TODO(@junrushao1994): After adding cuda device api, uncomment the line below.
  # list(APPEND MNM_SRC_FILES ${CMAKE_SOURCE_DIR}/src/device_api/cuda/*.cc)

  if (MNM_USE_CUDNN)
    file(GLOB_RECURSE CUDNN_SRCS ${CMAKE_SOURCE_DIR}/src/op/backend/cudnn/*.cc)
    list(APPEND MNM_SRC_FILES ${CUDNN_SRCS})
    list(APPEND MNM_LINK_LIBRARIES ${CUDA_CUDNN_LIBRARY})
    list(APPEND MNM_3RD_PARTY_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
  endif()

  list(APPEND MNM_LINK_LIBRARIES ${CUDA_CUDA_LIBRARY})
  list(APPEND MNM_LINK_LIBRARIES ${CUDA_CUDART_LIBRARY})

  list(APPEND MNM_3RD_PARTY_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)

endif()
