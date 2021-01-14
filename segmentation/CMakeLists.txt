add_library(segmentation inference.cc)
include_directories(${CMAKE_SOURCE_DIR})
target_link_libraries(segmentation PUBLIC ${TORCH_LIBRARIES} ${OpenCV_LIBS})