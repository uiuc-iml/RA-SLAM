add_library(segmentation inference.cc)
target_link_libraries(segmentation PUBLIC ${TORCH_LIBRARIES} ${OpenCV_LIBS})