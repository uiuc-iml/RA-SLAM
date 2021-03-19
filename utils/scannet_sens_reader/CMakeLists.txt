add_library(scannet_sens_reader scannet_sens_reader.cc)
target_link_libraries(scannet_sens_reader PUBLIC cuda_util ${OPENCV_LIBS} scannet)
