#include "ros_test.h"

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
#include <sensor_msgs/image_encodings.h>
#include <shape_msgs/Mesh.h>
#include <shape_msgs/MeshTriangle.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/String.h>

Test::Test() {
  std::string vocab_file_path;
  std::string config_file_path;
  std::string device_id;

  mNh.getParam("/ros_disinf_slam/calib_path", config_file_path); 
  mNh.getParam("/ros_disinf_slam/orb_vocab_path", vocab_file_path);

  meshPub = mNh.advertise<shape_msgs::Mesh>("/mesh", 1);
  ros::ServiceServer meshServ = mNh.advertiseService("/meshserv", &Test::serve_mesh, this);

  // visual_tools_.reset(new rviz_visual_tools::RvizVisualTools("world","/mesh_visual", mNh));
  // visual_tools_->setPsychedelicMode(false);
  // visual_tools_->loadMarkerPub();

  std::shared_ptr<openvslam::config> cfg;
  try {
    cfg = GetAndSetConfig(config_file_path);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return;
  }

  // Initialize cameras
  zed_native = std::make_shared<ZEDNative>(*cfg, 0);
  l515 = std::make_shared<L515>();
  // l515->SetDepthSensorOption(rs2_option::RS2_OPTION_CONFIDENCE_THRESHOLD, 3);
  // l515->SetDepthSensorOption(rs2_option::RS2_OPTION_LASER_POWER, 100);
  // l515->SetDepthSensorOption(rs2_option::RS2_OPTION_DIGITAL_GAIN, 1);

  // TODO Initialize poses

  // Initialize TSDF
  TSDF = std::make_shared<TSDFSystem>(
    CELL_SIZE,
    CELL_SIZE * 6,
    4,
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
  float minValue        = 0, maxValue = 0;
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

  // meshPub.publish(mesh_msg);
  // visual_tools_->publishMesh(*mesh_msg, rviz_visual_tools::ORANGE, 1, "mesh", 1); // rviz_visual_tools::TRANSLUCENT_LIGHT
  // Don't forget to trigger the publisher!
  // visual_tools_->trigger();
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
      if (SLAM->terminate_is_requested()) break;
      // get sensor readings
      const int64_t timestamp = zed_native->GetStereoFrame(&img_left, &img_right);

      // DEBUG
      cv::imshow("stereo", img_left);
      cv::waitKey(1);

      // visual slam
      const pose_valid_tuple m = SLAM->feed_stereo_images_w_feedback(img_left, img_right, timestamp / 1e3);
      const SE3<float> posecam_P_world(m.first.cast<float>().eval());

      // TODO This should be fixed with camera calibration
      SE3<float> pose(posecam_P_world.GetR(), posecam_P_world.GetT() * 7.8);

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
    while (ros::ok()) {
      cv::Mat img_rgb, img_depth;
      if (SLAM->terminate_is_requested()) break;
      const int64_t timestamp = l515->GetRGBDFrame(&img_rgb, &img_depth);

      // DEBUG
      cv::imshow("depth", img_depth);
      cv::waitKey(1);

      const SE3<float> posecam_P_world = POSE_MANAGER->query_pose(timestamp);
      cv::resize(img_rgb, img_rgb, cv::Size(), .5, .5);
      cv::resize(img_depth, img_depth, cv::Size(), .5, .5);
      img_depth.convertTo(img_depth, CV_32FC1, 1. / l515->DepthScale());
      TSDF->Integrate(posecam_P_world, img_rgb, img_depth);

      ros::spinOnce();
    }
  });

  // TODO Not sure if pose thread is still necessary

  t_slam.join();
  t_tsdf.join();
  SLAM->shutdown();
}

