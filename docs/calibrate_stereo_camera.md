# Calibrating camera intrinsics

For high-precision visual odometry application, it is often required to calibrate intrinsics
of cameras, which is a set of numbers that vary on a per-camera basis (e.g., two ZED stereo cameras
of same model may have vastly different intrinsics). This doc will not cover mathematical details
of camera intrinsics and how calibration is done, but simply provide a walk through of how to do it
for SLAM applications. (If you are interested, you can read more about it [here](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html))

We will use ZED camera as an example.

1. go to python_utils/

2. Invoke `python3 compute_zed_intrinsics.py`. You should see a video stream pops up. If not, please verify that
the ZED camera is properly plugged in. If you have multiple cameras setup, you may want to change the parameter passed
to main until you find the correct camera.

3. Print on a paper or use a monitor screen to display this [chessboard pattern](https://raw.githubusercontent.com/opencv/opencv/4.x/doc/pattern.png).

4. Hold the camera and point it towards the chessboard pattern. Make sure that both left and right camera of ZED can capture the entire chessboard. Press `s` to save view of the chessboard. Repeat this process for multiple times from different viewangles.

5. After you are done capturing chessboard (I typically use >8 images from different angles), press `q`. The camera intrinsics
matrix and undistortion coefficients should pop up on your screen. Here is an example output from my setup:

```
Camera matrix : 
array([[354.05822453, 0.          , 343.61677793],
       [  0.        , 353.74858221, 198.55061607],
       [  0.        ,   0.        ,   1.        ]])

dist :
array([[-0.16332529, -0.0248084 , -0.00068774, 0.00058966, 0.05670311]])
```

6. Update YAML config file accordingly. For instance, the above output would translate to the following YAML config.

```
Calibration.left.fx: 354.05822453
Calibration.left.fy: 353.74858221
Calibration.left.cx: 343.61677793
Calibration.left.cy: 198.55061607
Calibration.left.distortion: [-0.16332529, -0.0248084 , -0.00068774, 0.00058966, 0.05670311]
```
