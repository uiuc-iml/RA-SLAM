#include <ros/ros.h>
#include <spdlog/spdlog.h>

#include "ros_interface.h"
#include "ros_test.h"

/*
RosInterface is the existing one ported from Yu.
However, I had some issues getting it to work well,
there were some issues with slow tracking and updating.
So the current one I am working on is Test.

Rebuilt with `catkin build`.
Setup environment with `source devel/setup.bash`
Run with `roslaunch semantic_reconstruction disinfslam.launch

Due to some jankiness, startup procedure is weird: to be fixed
Start redrun, since we need redis server to be started
Immediately start ra_slam, otherwise trina will acquire the camera first and things will fail

There is still a bunch to do:
Correcting pose using base pose
Confidence threshold for each individual point
Fix issue with close points -> Seem to be a TSDF issue
*/

int main(int argc, char* argv[])
{
    // spdlog::set_level(spdlog::level::debug);
    ros::init(argc, argv, "ros_disinf_slam");
    // RosInterface rosInterface;
    Test test;
    ros::spin();
    return EXIT_SUCCESS;
}