## Installation
1. Follow the instructions on the `README.md` on the `main` branch.
2. Some additional packages are required: [`nlohmann/json`](https://github.com/nlohmann/json) and [`redis++`](https://github.com/sewenew/redis-plus-plus).
3. Create a [catkin workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace).
4. `git clone` this repository (`trina_ros` branch) into your workspace's `src` directory.

## Running
0. If you are running on the `TRINA` PC, this currently resides in `~/<my name>_ws`.
1. Run `catkin build` to build the files. You may need to run `source <ws>/devel/setup.bash`. `catkin clean` is a useful command if you need to reset your workspace (does not affect `src` files).
2. In `~/TRINA`, run `redrun redrun_config.json`.
2. Run `roslaunch semantic_reconstruction disinfslam.launch`. Refer to `examples/ros_camera_driver/main.cc` for the nuances.
3. There are a few errors that might occur here, mainly that the cameras may not be connected properly. Use `realsense-viewer` for the L515 and `ZED_Explorer` for the ZED2 to verify they are connected properly. The ZED2 camera is fairly finicky. We are currently using the `devid` in `launch/disinfslam.launch` (which is the same as `/dev/videoX`). May require some unplug/replugs. Note that we are using another ZEDMini as the eyes, so make sure that `RA_SLAM` is using the slam ZED2.
4. Once the cameras have started up, you will want to rotate `Trina` a few degrees to aid in finding initiaization keypoints. You should start seeing a steady stream of pose values in the terminal, followed by a `TSDF Started!` message. If the message appears before you are properly tracking, the `TSDF` may get messy. In that case, either move `Trina` more/start moving earlier, or increase the delay between finding tracking and starting `t_tsdf`.
5. There are 3 main threads:
    - `t_slam`, which continuously feeds frames to the `SLAM` module and retrives a pose
    - `t_tsdf`, which continuously adds points from the L515 depth image to the TSDF
    - `t_base_pose`, which continuously reads the `redis` server to retrieve the base pose, and records it for correcting the slam pose
5. Call `rosservice call /meshserv test` to retrieve the current world mesh. The last argument `test` is currently unused, it just requires some string value. There is a utility script `python_utils/update.bash` which polls for the mesh every 1 second.
8. Start `rviz` (requires `roscore` to be running). In `File/recent`, select `slam.rviz`, which should start populating the mesh. Alternatively, if the `VIS` setting in `SlamModule` in `TRINA` is set to `True`, the Klampt visualization should automatically pop up.

## Development
There are 2 main components.
1. The `ROS` side of things, which generates the mesh, and
2. The `TRINA` side of things, which consume the mesh.

### ROS
This is responsible for tracking the camera pose, adding points to the TSDF, and generating the mesh when queried.

### TRINA
I have implemented a basic module that exposes a function to check for collisions with specific links. More details are in the module file itself. It looks like it is implemented twice so as to run it on a separate process (as it consumes a lot of resources). Ask Patrick about it.

### Communication
[ROS Service](http://wiki.ros.org/Services) is used for communication. The documentation is fairly instructive. However note that you will have to copy the `Python` files from `devel/lib/python3/dist-packages/semantic_reconstruction` into a similar folder in `TRINA` (there should already be one).

## TODO
There is a bunch of improvements that can be made.
0. Calibrate the extrinsics of the camera, and the intrinsics.
1. For some reason, points closer than around 0.5m to the camera will not be added to the TSDF (I'm pretty sure they are detected by the L515).
2. Make masking work again. Changes to KrisLibrary has broken some stuff, you may need to rewrite some portions of the `mask_robot` library (or reinstall a previous version of KrislLibrary). Ask Patrick for more information.
2. Figuring out how to use the robot pose to reduce slam pose estimation error. The basic structure is already in place, just need to figure out how to find the difference matrix. This would also require making some changes to the robot state, to include a last updated timestamp.
3. Integrate with Dynamic Mesh Server. However, I'm not exactly sure if this is required - you can already query the TSDF for a specific region of the world. It is currently hardcoded to some bounding box `bbox`, but one can definitely add arguments to `meshsrv` to specify the bounding box (or maybe even just create a bounding box centered around the pose).
4. There are quite a few TODOs lying around, classified as MAJOR, EASY etc.