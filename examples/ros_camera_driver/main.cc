#include <ros/ros.h>

#include "ros_interface.h"

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "ros_disinf_slam");
    RosInterface rosInterface;
    ros::spin();
}