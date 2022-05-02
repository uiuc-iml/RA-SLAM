import pyzed.sl as sl

zed = sl.Camera()
init_params = sl.InitParameters()
sl.camera_resolution = sl.RESOLUTION.VGA
sl.camera_fps = 100
zed.open(init_params)
calib_params = zed.get_camera_information().camera_configuration.calibration_parameters
print(calib_params.left_cam.fx)
print(calib_params.left_cam.fy)
print(calib_params.left_cam.cx)
print(calib_params.left_cam.cy)
print(calib_params.left_cam.disto)
print(calib_params.right_cam.fx)
print(calib_params.right_cam.fy)
print(calib_params.right_cam.cx)
print(calib_params.right_cam.cy)
print(calib_params.right_cam.disto)
print(calib_params.T)
print(calib_params.R)