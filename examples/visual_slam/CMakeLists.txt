add_executable(run_sr300 run_sr300.cc)
target_link_libraries(run_sr300 PRIVATE sr300 modules)

add_executable(run_zed_native run_zed_native.cc)
target_link_libraries(run_zed_native PRIVATE zed_native modules)

add_executable(run_zed_native_l515 run_zed_native_l515.cc)
target_link_libraries(run_zed_native_l515 PRIVATE zed_native l515 modules)

if (${ZED_FOUND})
    add_executable(run_zed run_zed.cc)
    target_link_libraries(run_zed PRIVATE zed modules)
endif (${ZED_FOUND})
