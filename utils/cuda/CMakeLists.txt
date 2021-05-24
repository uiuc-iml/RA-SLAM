add_library(cuda_util INTERFACE)
target_link_libraries(cuda_util INTERFACE Eigen3::Eigen ${CUDA_LIBRARIES})
target_include_directories(cuda_util SYSTEM INTERFACE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
# TODO(alvin): remove this when Eigen is upgraded
target_compile_options(cuda_util INTERFACE -Wno-cpp)

