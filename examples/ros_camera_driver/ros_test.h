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

// Retrieving poses
#include <sw/redis++/redis++.h>

using namespace sw::redis;

#define TRUNCATION_DISTANCE -0.1
#define CELL_SIZE 0.05
#define MAX_DEPTH 2.5

/*
 * TODOs
 * 1. Calibrate ZED Cameras
 * 2. Figure out why Ctrl^C is not killing this
 * 3. Use TRINA pose to fix pose estimations
 * 4. Figure out how to identify points faster.
 * 	It may be a procedure thing, show an interesting scene
 * 	and strafe TRINA
 */

class Test {
public:
  Test();

  bool serve_mesh(semantic_reconstruction::Mesh::Request& request, semantic_reconstruction::Mesh::Response& response);

private:
  Redis redis = Redis("tcp://127.0.0.1:6379"); // TODO Should not be hardcoded

  ros::NodeHandle mNh;
  rviz_visual_tools::RvizVisualToolsPtr visual_tools_;

  tf2_ros::Buffer tfBuffer;
  Eigen::Isometry3d world_T_slam;

  // Cameras
  std::shared_ptr<ZEDNative>	zed_native;
  std::shared_ptr<L515>		l515;

  // Systems
  bool is_tracking; // TODO Needs a mutex

  std::shared_ptr<SLAMSystem>	SLAM;
  std::shared_ptr<TSDFSystem>	TSDF;
  std::shared_ptr<pose_manager>	POSE_MANAGER;

  // Output Mesh
  shape_msgs::Mesh::Ptr mesh_msg;

  void reconstruct();
  void generate_mesh();
};
