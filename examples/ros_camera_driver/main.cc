#include <ros/ros.h>
#include <spdlog/spdlog.h>

#include "ros_interface.h"
#include "ros_test.h"

int main(int argc, char* argv[])
{
    spdlog::set_level(spdlog::level::debug);
    ros::init(argc, argv, "ros_disinf_slam");
    // RosInterface rosInterface;
    Test test;
    ros::spin();
    return EXIT_SUCCESS;
}