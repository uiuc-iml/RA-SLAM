#include <ros/ros.h>
#include <openvslam/publish/map_publisher.h>
#include <openvslam/system.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <iostream>
#include <popl.hpp>
#include <string>
#include <thread>

#include <rviz_visual_tools/rviz_visual_tools.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2/LinearMath/Transform.h>

#include "cameras/l515.h"
#include "cameras/zed_native.h"
#include "modules/renderer_module.h"
#include "modules/slam_module.h"
#include "modules/tsdf_module.h"
#include "utils/config_reader.hpp"
#include "utils/cuda/errors.cuh"
#include "utils/gl/renderer_base.h"
#include "utils/rotation_math/pose_manager.h"
#include "utils/time.hpp"
#include "semantic_reconstruction/Mesh.h"

#define TRUNCATION_DISTANCE -0.1
#define CELL_SIZE 0.05

class Test {
public:
    Test();

    void reconstruct();

    void generate_mesh();

    bool serve_mesh(semantic_reconstruction::Mesh::Request& request, semantic_reconstruction::Mesh::Response& response);

private:
    ros::NodeHandle mNh;
    ros::Publisher meshPub;
    // rviz_visual_tools::RvizVisualToolsPtr visual_tools_;

    // tf2_ros::Buffer tfBuffer;
    // Eigen::Isometry3d T_ws;

    std::shared_ptr<ZEDNative> zed_native;
    std::shared_ptr<L515> l515;

    std::shared_ptr<SLAMSystem> SLAM;
    std::shared_ptr<TSDFSystem> TSDF;
    std::shared_ptr<pose_manager> POSE_MANAGER;

    shape_msgs::Mesh::Ptr mesh_msg;
};
