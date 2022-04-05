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
    // initialize cameras
    zed_native = std::make_shared<ZEDNative>(*cfg, 0);
    l515 = std::make_shared<L515>();
    // l515->SetDepthSensorOption(rs2_option::RS2_OPTION_CONFIDENCE_THRESHOLD, 3);
    // l515->SetDepthSensorOption(rs2_option::RS2_OPTION_LASER_POWER, 100);
    // l515->SetDepthSensorOption(rs2_option::RS2_OPTION_DIGITAL_GAIN, 1);

    // tfListener = std::make_shared<tf2_ros::TransformListener>(tfBuffer);
    // geometry_msgs::TransformStamped transformStampedInit = tfBuffer.lookupTransform("world", "slam", ros::Time::now(), ros::Duration(5));
    // T_ws = tf2::transformToEigen(transformStampedInit);

    // initialize slam
    SLAM = std::make_shared<SLAMSystem>(cfg, vocab_file_path);

    // initialize TSDF
    TSDF = std::make_shared<TSDFSystem>(CELL_SIZE, CELL_SIZE * 6, 4, GetIntrinsicsFromFile(config_file_path),
                                            GetExtrinsicsFromFile(config_file_path));
    SLAM->startup();

    POSE_MANAGER = std::make_shared<pose_manager>();

    reconstruct();
}

void Test::generate_mesh()
{
    // std::cout << "=====================" << std::endl;
    static BoundingCube<float> volumn = { -4, 4, -4, 4, -8, 8 };

    auto SemanticReconstr   = TSDF->Query(volumn);
    int numPoints           = SemanticReconstr.size();
    float minValue          = 1e100, maxValue = -1e100;
    Math3D::AABB3D bbox;

    for (int i = 0; i < numPoints; i++) {
        minValue = Min(minValue,SemanticReconstr[i].tsdf);
        maxValue = Max(maxValue,SemanticReconstr[i].tsdf);
        bbox.expand(Math3D::Vector3(SemanticReconstr[i].position[0],SemanticReconstr[i].position[1],SemanticReconstr[i].position[2]));
    }

    printf("Read %d points with distance in range [%g,%g]\n", numPoints, minValue, maxValue);

    float truncation_distance = TRUNCATION_DISTANCE;
    if (TRUNCATION_DISTANCE < 0) {
        //auto-detect truncation distance
        truncation_distance = Max(-minValue,maxValue)*0.99;
    }

    Geometry::SparseTSDFReconstruction tsdf(Math3D::Vector3(CELL_SIZE), truncation_distance);
    tsdf.tsdf.defaultValue[0] = truncation_distance;
    Math3D::Vector3 ofs(CELL_SIZE*0.5);
    for (int i = 0; i < numPoints; i++) {
        tsdf.tsdf.SetValue(Math3D::Vector3(SemanticReconstr[i].position[0],SemanticReconstr[i].position[1],SemanticReconstr[i].position[2]) + ofs,SemanticReconstr[i].tsdf);
    }

    Meshing::TriMesh mesh;
    tsdf.ExtractMesh(mesh);
    // // std::cout<<"Before Merge: trisSize: "<<mesh.tris.size()<<std::endl;
    MergeVertices(mesh, 0.05);

    int vertsSize = mesh.verts.size();
    int trisSize = mesh.tris.size();
    // std::cout << "trisSize: " << trisSize << std::endl;

    mesh_msg = boost::make_shared<shape_msgs::Mesh>();
    // geometry_msgs/Point[]
    mesh_msg->vertices.resize(vertsSize);
    // shape_msgs/MeshTriangle[]
    mesh_msg->triangles.resize(trisSize);

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

bool Test::serve_mesh(semantic_reconstruction::Mesh::Request& request, semantic_reconstruction::Mesh::Response& response) {
  generate_mesh();
  response.value = *mesh_msg;
  return true;
}

void Test::reconstruct() {

//   ImageRenderer renderer("tsdf", std::bind(&pose_manager::get_latest_pose, POSE_MANAGER), TSDF,
//                          GetIntrinsicsFromFile(config_file_path));

  std::thread t_slam([&]() {
    while (ros::ok()) {
      cv::Mat img_left, img_right;
      if (SLAM->terminate_is_requested()) break;
      // get sensor readings
      const int64_t timestamp = zed_native->GetStereoFrame(&img_left, &img_right);
      cv::imshow("stereo", img_left);
      cv::waitKey(1);
      // visual slam
      const pose_valid_tuple m =
          SLAM->feed_stereo_images_w_feedback(img_left, img_right, timestamp / 1e3);
      const SE3<float> posecam_P_world(m.first.cast<float>().eval());
      SE3<float> pose(posecam_P_world.GetR(), posecam_P_world.GetT() * 7.8);
      if (m.second) POSE_MANAGER->register_valid_pose(timestamp, pose);
      ros::spinOnce();
    }
  });

  std::thread t_tsdf([&]() {
    const auto map_publisher = SLAM->get_map_publisher();
    while (ros::ok()) {
        cv::Mat img_rgb, img_depth;
        if (SLAM->terminate_is_requested()) break;
        const int64_t timestamp = l515->GetRGBDFrame(&img_rgb, &img_depth);
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

//   renderer.Run();
  t_slam.join();
  t_tsdf.join();
  SLAM->shutdown();
}