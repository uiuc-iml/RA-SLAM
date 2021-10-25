#include <chrono>
#include <cinttypes>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>

#include <cv_bridge/cv_bridge.h>
#include <openvslam/system.h>
#include <popl.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <image_transport/image_transport.h>
#include <KrisLibrary/geometry/TSDFReconstruction.h>
#include <KrisLibrary/math3d/AABB3D.h>
#include <KrisLibrary/meshing/IO.h>
#include <KrisLibrary/meshing/TriMeshOperators.h>
#include <KrisLibrary/utils.h>
#include <KrisLibrary/utils/ioutils.h>
#include <ros/ros.h>
#include <rviz_visual_tools/rviz_visual_tools.h>
#include <sensor_msgs/image_encodings.h>
#include <shape_msgs/Mesh.h>
#include <shape_msgs/MeshTriangle.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/String.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/convert.h>
#include <tf2/LinearMath/Transform.h>

#include "cameras/l515.h"
#include "cameras/zed_native.h"
#include "disinfect_slam/disinfect_slam.h"
#include "utils/config_reader.hpp"
#include "utils/time.hpp"

#define CELL_SIZE 0.01
#define TRUNCATION_DISTANCE -0.1
using namespace std;

class RosInterface
{
public:
  RosInterface();
  void publishImage(cv_bridge::CvImagePtr & imgBrgPtr, const cv::Mat & img, ros::Publisher & pubImg, std::string imgFrameId, std::string dataType, ros::Time t);
  void reconstTimerCallback(const ros::TimerEvent&);
  void poseTimerCallback(const ros::TimerEvent&);
  void tsdfCb(std::vector<VoxelSpatialTSDF> & SemanticReconstr);
  void run();
  void zedMaskCb(const sensor_msgs::ImageConstPtr& msg);
  void l515MaskCb(const sensor_msgs::ImageConstPtr& msg);

private:
  ros::NodeHandle mNh;
  std::string model_path, calib_path, orb_vocab_path;
  int devid;
  bool use_mask, require_mesh, global_mesh, renderFlag;
  double bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max, bbox_z_min, bbox_z_max;
  Eigen::Isometry3d T_ws;
  // geometry_msgs::Pose T_ws;
  geometry_msgs::TransformStamped transformStamped;
  image_transport::Subscriber maskLeft;
  image_transport::Subscriber maskDepth;

  tf2_ros::TransformBroadcaster mTfSlam;
  ros::Publisher mPubL515RGB ;
  ros::Publisher mPubL515Depth;
  ros::Publisher mPubZEDImgL;
  ros::Publisher mPubZEDImgR;
  // cv bridges
  cv_bridge::CvImagePtr mL515RGBBrg;
  cv_bridge::CvImagePtr mL515DepthBrg;
  cv_bridge::CvImagePtr mZEDImgLBrg;
  cv_bridge::CvImagePtr mZEDImgRBrg;
  // initialize slam
  ros::Timer reconstTimer;
  ros::Timer poseTimer;

  cv::Mat zedLeftMask;
  cv::Mat l515Mask;
  std::mutex mask_lock;
  std::mutex zed_mask_lock;

  rviz_visual_tools::RvizVisualToolsPtr visual_tools_;
  tf2_ros::Buffer tfBuffer;
  ros::Publisher meshPub;
  std::shared_ptr<tf2_ros::TransformListener> tfListener;

  std::shared_ptr<DISINFSystem> my_sys;
  std::shared_ptr<ZEDNative> zed_native;
  std::shared_ptr<L515> l515;
};
