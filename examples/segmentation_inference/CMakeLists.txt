if (CMAKE_CUDA_COMPILER)
    add_executable(sample_inference main.cc)
    target_link_libraries(sample_inference
        PRIVATE segmentation ${OpenCV_LIBS})
endif ()
