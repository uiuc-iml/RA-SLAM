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