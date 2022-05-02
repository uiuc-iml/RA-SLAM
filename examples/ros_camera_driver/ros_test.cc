#include <nlohmann/json.hpp> // This has to be here. Some included file has `using namespace std`, which will conflict with this file

#include "ros_test.h"

#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <image_transport/image_transport.h>
#include <KrisLibrary/geometry/TSDFReconstruction.h>
#include <KrisLibrary/math3d/AABB3D.h>
#include <KrisLibrary/math3d/rotation.h>
#include <KrisLibrary/meshing/IO.h>
#include <KrisLibrary/meshing/TriMeshOperators.h>
#include <KrisLibrary/utils.h>
#include <KrisLibrary/utils/ioutils.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <shape_msgs/Mesh.h>
#include <shape_msgs/MeshTriangle.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/String.h>
#include <yaml-cpp/yaml.h>

using json = nlohmann::json;

Test::Test() {
  diff_pose = SE3<float>::Identity();

  std::string vocab_file_path;
  std::string config_file_path;
  std::string device_id;
  int devid;
  float max_depth;

  mNh.getParam("/ros_disinf_slam/calib_path", config_file_path); 
  mNh.getParam("/ros_disinf_slam/orb_vocab_path", vocab_file_path);
  mNh.getParam("/ros_disinf_slam/devid", devid);
  mNh.getParam("/ros_disinf_slam/max_depth", max_depth);

  ros::ServiceServer meshServ = mNh.advertiseService("/meshserv", &Test::serve_mesh, this);

  auto tfListener = std::make_shared<tf2_ros::TransformListener>(tfBuffer);
  geometry_msgs::TransformStamped transformStampedInit = tfBuffer.lookupTransform("world", "slam", ros::Time::now(), ros::Duration(5));
  world_T_slam = tf2::transformToEigen(transformStampedInit);

  visual_tools_.reset(new rviz_visual_tools::RvizVisualTools("world","/mesh_visual", mNh));
  visual_tools_->setPsychedelicMode(false);
  visual_tools_->loadMarkerPub();

  std::shared_ptr<openvslam::config> cfg;
  try {
    cfg = GetAndSetConfig(config_file_path);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return;
  }

  // Initialize cameras
  zed_native = std::make_shared<ZEDNative>(*cfg, devid);
  l515 = std::make_shared<L515>();
  YAML::Node yaml_node = YAML::LoadFile(config_file_path);
  l515->SetDepthSensorOption(rs2_option::RS2_OPTION_DIGITAL_GAIN, yaml_node["L515.DigitalGain"].as<int>());                 // [0,1]
  l515->SetDepthSensorOption(rs2_option::RS2_OPTION_LASER_POWER, yaml_node["L515.LaserPower"].as<int>());                   // [0,100]
  l515->SetDepthSensorOption(rs2_option::RS2_OPTION_CONFIDENCE_THRESHOLD, yaml_node["L515.ConfidenceThreshold"].as<int>()); // [0,3]
  l515->SetDepthSensorOption(rs2_option::RS2_OPTION_MIN_DISTANCE, yaml_node["L515.MinDistance"].as<int>());
  l515->SetDepthSensorOption(rs2_option::RS2_OPTION_POST_PROCESSING_SHARPENING, yaml_node["L515.PostProcSharp"].as<int>()); // [0,3]
  l515->SetDepthSensorOption(rs2_option::RS2_OPTION_PRE_PROCESSING_SHARPENING, yaml_node["L515.PreProcSharp"].as<int>());   // [0,5]
  l515->SetDepthSensorOption(rs2_option::RS2_OPTION_NOISE_FILTERING, yaml_node["L515.NoiseFiltering"].as<int>());           // [0,6]

  // TODO Initialize poses

  // Initialize TSDF
  TSDF = std::make_shared<TSDFSystem>(
    CELL_SIZE,
    CELL_SIZE * 6,
    max_depth,
    GetIntrinsicsFromFile(config_file_path),
    GetExtrinsicsFromFile(config_file_path)
  );

  // Initialize SLAM
  SLAM = std::make_shared<SLAMSystem>(cfg, vocab_file_path);
  SLAM->startup();

  // Initialize Pose
  POSE_MANAGER = std::make_shared<pose_manager>();

  // Start reconstruction loop
  reconstruct();
}

bool Test::serve_mesh(semantic_reconstruction::Mesh::Request& request, semantic_reconstruction::Mesh::Response& response) {
  generate_mesh();
  response.value = *mesh_msg;
  return true;
}

/*
 * Generates Mesh from TSDF within some hardcoded region around the world origin
 */
void Test::generate_mesh() {
  // Expanding Bounding Box of TSDF 
  static BoundingCube<float> volume = { -4, 4, -4, 4, -8, 8 };
  auto reconstruction	= TSDF->Query(volume);
  int numPoints		= reconstruction.size();
  float minValue        = 100, maxValue = -100;
  Math3D::AABB3D bbox;

  for (int i = 0; i < numPoints; i++) {
    const auto& point	= reconstruction[i];
    minValue = Min(minValue, point.tsdf);
    maxValue = Max(maxValue, point.tsdf);
    bbox.expand(Math3D::Vector3(point.position[0], point.position[1], point.position[2]));
  }

  printf("Read %d points with distance in range [%g,%g]\n", numPoints, minValue, maxValue);

  float truncation_distance = TRUNCATION_DISTANCE;
  if (truncation_distance < 0) {
    //auto-detect truncation distance
    truncation_distance = Max(-minValue, maxValue) * 0.99;
  }

  // Class Conversion
  Geometry::SparseTSDFReconstruction tsdf(Math3D::Vector3(CELL_SIZE), truncation_distance);
  tsdf.tsdf.defaultValue[0] = truncation_distance;
  Math3D::Vector3 offset(CELL_SIZE * 0.5);

  for (int i = 0; i < numPoints; i++) {
    const auto& point = reconstruction[i];
    tsdf.tsdf.SetValue(Math3D::Vector3(
      point.position[0],
      point.position[1],
      point.position[2]) + offset,
      point.tsdf
    );
  }

  // Convert TSDF to Mesh
  Meshing::TriMesh mesh;
  tsdf.ExtractMesh(mesh);
  MergeVertices(mesh, 0.05);

  int vertsSize	= mesh.verts.size();
  int trisSize	= mesh.tris.size();
  // std::cout << "trisSize: " << trisSize << std::endl;

  // Construct Mesh message
  mesh_msg = boost::make_shared<shape_msgs::Mesh>();
  mesh_msg->vertices.resize(vertsSize); // geometry_msgs/Point[]
  mesh_msg->triangles.resize(trisSize); // shape_msgs/MeshTriangle[]

  for (int i = 0; i < vertsSize; i++) {
    mesh_msg->vertices[i].x = mesh.verts[i].x;
    mesh_msg->vertices[i].y = mesh.verts[i].y;
    mesh_msg->vertices[i].z = mesh.verts[i].z;
    // std::cout<<mesh.verts[i].x<<std::endl;
  }

  for(int i = 0; i < trisSize; i++){
    mesh_msg->triangles[i].vertex_indices[0] = mesh.tris[i].a;
    mesh_msg->triangles[i].vertex_indices[1] = mesh.tris[i].b;
    mesh_msg->triangles[i].vertex_indices[2] = mesh.tris[i].c;
    // std::cout<<mesh.tris[i].a<<std::endl;
  }

  visual_tools_->publishMesh(world_T_slam, *mesh_msg, rviz_visual_tools::ORANGE, 1, "mesh", 1);
  // Don't forget to trigger the publisher!
  visual_tools_->trigger();
}

void Test::reconstruct() {
  /*
   * In each loop,
   * 1. Retrieves the stereo images and the corresponding timestamp
   * 2. Obtains a pose estimation from sending the images to the SLAM system
   * 3. Registers the new pose with the POSE_MANAGER
   */
  std::thread t_slam([&]() {
    while (ros::ok()) {
      cv::Mat img_left, img_right;
      // get sensor readings
      const int64_t timestamp = zed_native->GetStereoFrame(&img_left, &img_right);

      // DEBUG
      cv::imshow("stereo", img_left);
      cv::waitKey(1);

      // visual slam
      const pose_valid_tuple m = SLAM->feed_stereo_images_w_feedback(img_left, img_right, timestamp * 1e3); // TODO Unsure units
      const SE3<float> posecam_P_world(m.first.cast<float>().eval());

      // TODO This should be fixed with camera calibration
      SE3<float> pose(posecam_P_world.GetR(), posecam_P_world.GetT() * 10);
      slam_pose = pose; // TODO Medium we will have to update robot_state to return time last updated
      pose = diff_pose * pose;
      // pose = SE3<float>(pose.GetR() * diff_pose.GetR(), pose.GetT() + pose.GetT());

      if (has_started) {
        auto euler = pose.GetR().toRotationMatrix().eulerAngles(0, 1, 2);
        std::cout << std::setw(5) << "|T: "
            << std::setprecision(3)
            << std::setw(12) << pose.GetT().x() << " "
            << std::setw(12) << pose.GetT().y() << " "
            << std::setw(12) << pose.GetT().z() << " "
            << std::setw(12) << euler[0] << " "
            << std::setw(12) << euler[1] << " "
            << std::setw(12) << euler[2] << " "
            << std::endl;
      }

      is_tracking = m.second;
      if (m.second) POSE_MANAGER->register_valid_pose(timestamp, pose);
      ros::spinOnce();
    }
  });

  /*
   * In each loop,
   * 1. Retrieve depth image and corresponding timestamp
   * 2. Obtain estimated pose from POSE_MANAGER corresponding to the timestamp
   * 3. Scale and integrate the depth image at that pose into the TSDF
   */
  std::thread t_tsdf([&]() {
    has_started = false;
    static int64_t start_time = 0;
    while (ros::ok()) {
      cv::Mat img_rgb, img_depth;
      const int64_t timestamp = l515->GetRGBDFrame(&img_rgb, &img_depth);

      // if (POSE_MANAGER->query_pose(timestamp) == SE3<float>::Identity()) {
      if (!is_tracking) {
        continue;
      }

      if (start_time == 0) {
        // TODO should have a better strategy, e.g. wait for no tracking loss in 5 s
        start_time = GetTimestamp<std::chrono::milliseconds>() + 5000; // Wait 3s after acquiring
        continue;
      }

      if (GetTimestamp<std::chrono::milliseconds>() < start_time) {
        continue;
      }
      
      if (!has_started) {
        has_started = true;
        std::cout << "TSDF started!" << std::endl;
      }

      // DEBUG
      // cv::imshow("depth", img_depth);
      // cv::waitKey(1);

      const SE3<float> posecam_P_world = POSE_MANAGER->query_pose(timestamp);
      cv::resize(img_rgb, img_rgb, cv::Size(), .5, .5);
      cv::resize(img_depth, img_depth, cv::Size(), .5, .5);
      img_depth.convertTo(img_depth, CV_32FC1, 1. / l515->DepthScale());
      TSDF->Integrate(posecam_P_world, img_rgb, img_depth);

      ros::spinOnce();
    }
  });

  ros::Rate rate(30);
  std::thread t_base_pose([&]() {
    while (ros::ok()) {
      if (!has_started) continue;

      auto t_val = redis.command<OptionalString>("JSON.GET", "ROBOT_STATE");
      if (t_val) {
        // std::cout << *t_val << std::endl;
        auto json_q = json::parse(*t_val)["Position"]["Robotq"];
        Eigen::Vector3f trans(json_q[1], json_q[2], - (float) json_q[0]);
        Eigen::Quaternionf rot(Eigen::AngleAxisf(json_q[3], Eigen::Vector3f(0, 1, 0))); // TODO MAJOR Will need to include the other axes, the ground may be sloped
        // std::cout << "BASE: " << json_q[0] << " " << json_q[1] << " " << json_q[2] << " " << json_q[3] << std::endl;
        SE3<float> base_pose(rot, trans);
        // diff_pose = SE3<float>(slam_pose.GetR().inverse() * base_pose.GetR(), base_pose.GetT() - slam_pose.GetT());
        diff_pose = slam_pose.Inverse() * base_pose; // TODO translation correction is wrong, but rotation at 0, 0 is correct

        // auto euler = diff_pose.GetR().toRotationMatrix().eulerAngles(0, 1, 2);
        // std::cout << std::setw(5) << "|T: "
        //     << std::setprecision(5)
        //     << std::setw(12) << diff_pose.GetT().x() << " "
        //     << std::setw(12) << diff_pose.GetT().y() << " "
        //     << std::setw(12) << diff_pose.GetT().z() << " "
        //     << std::setw(12) << euler[0] << " "
        //     << std::setw(12) << euler[1] << " "
        //     << std::setw(12) << euler[2] << " "
        //     << std::endl;
        // 0 is -z, forward, 1 is +x, left is positive, 2 is up down, 3 is rotation, left is positive
      }
      ros::spinOnce();
      rate.sleep();
    }
  });

  // TODO Not sure if pose thread is still necessary

  t_slam.join();
  t_tsdf.join();
  t_base_pose.join();
  SLAM->shutdown();
}

