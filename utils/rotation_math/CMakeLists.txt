add_library(rotation_math pose_manager.cc)
target_link_libraries(rotation_math PUBLIC cuda_util)
