ADD_SUBDIRECTORY(runtime_compiler)

FILE(GLOB curr_cxx_srcs *.cc)
FILE(GLOB curr_test_srcs *_test.cc)
FOREACH(test ${curr_test_srcs})
  LIST(REMOVE_ITEM curr_cxx_srcs ${test})
ENDFOREACH()

FILE(GLOB curr_cuda_srcs *.cu)
FILE(GLOB curr_test_cuda *_test.cu)
FOREACH(test ${curr_test_cuda})
  LIST(REMOVE_ITEM curr_cuda_srcs ${test})
ENDFOREACH()

SET(cxx_srcs ${cxx_srcs} ${curr_cxx_srcs} PARENT_SCOPE)
SET(cuda_srcs ${cuda_srcs} ${curr_cuda_srcs} PARENT_SCOPE)
SET(test_cxx_srcs ${test_cxx_srcs} ${curr_test_srcs} PARENT_SCOPE)
SET(test_cuda_srcs ${test_cuda_srcs} ${curr_test_cuda} PARENT_SCOPE)
