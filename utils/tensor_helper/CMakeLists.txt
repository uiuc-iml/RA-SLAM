add_library(tensor_helper libtorch_helper.cc)
target_link_libraries(tensor_helper PUBLIC ${TORCH_LIBRARIES} ${OpenCV_LIBS})
