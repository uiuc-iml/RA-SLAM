#include "ros_interface.h"

#include <iomanip>

RosInterface::RosInterface() {
  mNh.getParam("/ros_disinf_slam/model_path", model_path);
  mNh.getParam("/ros_disinf_slam/calib_path", calib_path); 
  mNh.getParam("/ros_disinf_slam/orb_vocab_path", orb_vocab_path);
  mNh.param("/ros_disinf_slam/devid", devid, 2);
  mNh.param("/ros_disinf_slam/bbox_x_min", bbox_x_min, -8.0);
  mNh.param("/ros_disinf_slam/bbox_x_max", bbox_x_max, 8.0);
  mNh.param("/ros_disinf_slam/bbox_y_min", bbox_y_min, -6.0);
  mNh.param("/ros_disinf_slam/bbox_y_max", bbox_y_max, 2.0);
  mNh.param("/ros_disinf_slam/bbox_z_min", bbox_z_min, -8.0);
  mNh.param("/ros_disinf_slam/bbox_z_max", bbox_z_max, 8.0);

  mNh.param("/ros_disinf_slam/renderer", renderFlag, false);
  mNh.param("/ros_disinf_slam/global_mesh", global_mesh, true);
  mNh.param("/ros_disinf_slam/require_mesh", require_mesh, true);

  auto cfg = GetAndSetConfig(calib_path);

  zed_native.reset(new ZEDNative(*cfg, devid));
  l515.reset(new L515());

  image_transport::ImageTransport it(mNh);
  maskLeft = it.subscribe("/robot_mask/zed_slam_left", 1, &RosInterface::zedMaskCb, this);
  maskDepth = it.subscribe("/robot_mask/realsense_slam_l515", 1, &RosInterface::l515MaskCb, this);
  meshPub = mNh.advertise<shape_msgs::Mesh>("/mesh", 1);
  mPubL515RGB = mNh.advertise<sensor_msgs::Image>("/l515_rgb", 1);
  mPubL515Depth = mNh.advertise<sensor_msgs::Image>("/l515_depth", 1);
  mPubZEDImgL = mNh.advertise<sensor_msgs::Image>("/zed_left_rgb", 1);
  mPubZEDImgR = mNh.advertise<sensor_msgs::Image>("/zed_right_rgb", 1);

  ros::ServiceServer meshServ = mNh.advertiseService("/meshserv", &RosInterface::meshServCb, this);

  // cv bridges
  mL515RGBBrg.reset(new cv_bridge::CvImage);
  mL515DepthBrg.reset(new cv_bridge::CvImage);
  mZEDImgLBrg.reset(new cv_bridge::CvImage);
  mZEDImgRBrg.reset(new cv_bridge::CvImage);

  my_sys   = std::make_shared<DISINFSystem>(calib_path, orb_vocab_path, model_path, CELL_SIZE, 0.2, 4, renderFlag);
  visual_tools_.reset(new rviz_visual_tools::RvizVisualTools("world","/mesh_visual", mNh));
  visual_tools_->setPsychedelicMode(false);
  visual_tools_->loadMarkerPub();
  tfListener= std::make_shared<tf2_ros::TransformListener>(tfBuffer);

  try{
      geometry_msgs::TransformStamped transformStampedInit = tfBuffer.lookupTransform("world", "slam", ros::Time::now(), ros::Duration(5));
      T_ws = tf2::transformToEigen(transformStampedInit);
      // T_ws.position.x = transformStampedInit.transform.translation.x;
      // T_ws.position.y = transformStampedInit.transform.translation.y;
      // T_ws.position.z = transformStampedInit.transform.translation.z;

      // T_ws.orientation =  transformStampedInit.transform.rotation;
      std::cout<<"Init world slam transform!"<<std::endl;
  }
  catch (tf2::TransformException &ex) {
      ROS_WARN("%s",ex.what());   
  }

  run();
}

bool RosInterface::meshServCb(semantic_reconstruction::Mesh::Request& request, semantic_reconstruction::Mesh::Response& response) {
  return true;
}

void RosInterface::publishImage(cv_bridge::CvImagePtr & imgBrgPtr, const cv::Mat & img, ros::Publisher & pubImg, std::string imgFrameId, std::string dataType, ros::Time t){
  imgBrgPtr->header.stamp = t;
  imgBrgPtr->header.frame_id = imgFrameId;
  imgBrgPtr->encoding = dataType;
  imgBrgPtr->image = img;
  pubImg.publish(imgBrgPtr->toImageMsg());
}

void RosInterface::tsdfCb(std::vector<VoxelSpatialTSDF> & SemanticReconstr)
{
    const auto st = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());
    int numPoints = SemanticReconstr.size();
    float minValue = 1e100, maxValue = -1e100;
    Math3D::AABB3D bbox;
    for(int i=0;i<numPoints;i++) {
        minValue = Min(minValue,SemanticReconstr[i].tsdf);
        maxValue = Max(maxValue,SemanticReconstr[i].tsdf);
        bbox.expand(Math3D::Vector3(SemanticReconstr[i].position[0],SemanticReconstr[i].position[1],SemanticReconstr[i].position[2]));
    }
    // printf("Read %d points with distance in range [%g,%g]\n",numPoints,minValue,maxValue);
    // printf("   x range [%g,%g]\n",bbox.bmin.x,bbox.bmax.x);
    // printf("   y range [%g,%g]\n",bbox.bmin.y,bbox.bmax.y);
    // printf("   z range [%g,%g]\n",bbox.bmin.z,bbox.bmax.z);
    float truncation_distance = TRUNCATION_DISTANCE;
    // float truncation_distance = 2.0*0.99;
    if(TRUNCATION_DISTANCE < 0) {
        //auto-detect truncation distance
        truncation_distance = Max(-minValue,maxValue)*0.99;
        // printf("Auto-detected truncation distance %g\n",truncation_distance);
    }
    // printf("Using cell size %g\n",CELL_SIZE);
    Geometry::SparseTSDFReconstruction tsdf(Math3D::Vector3(CELL_SIZE),truncation_distance);
    tsdf.tsdf.defaultValue[0] = truncation_distance;
    Math3D::Vector3 ofs(CELL_SIZE*0.5);
    for(int i=0;i<numPoints;i++) {
        tsdf.tsdf.SetValue(Math3D::Vector3(SemanticReconstr[i].position[0],SemanticReconstr[i].position[1],SemanticReconstr[i].position[2])+ofs,SemanticReconstr[i].tsdf);
    }

    Meshing::TriMesh mesh;
    tsdf.ExtractMesh(mesh);
    // std::cout<<"Before Merge: trisSize: "<<mesh.tris.size()<<std::endl;
    MergeVertices(mesh, 0.05);
    int vertsSize = mesh.verts.size();
    int trisSize = mesh.tris.size();
    std::cout<<"trisSize: "<<trisSize<<std::endl;
    const auto end = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());
    // std::cout<<"mesh processing time: "<<end-st<<" ms"<<std::endl;
    shape_msgs::Mesh::Ptr mMeshMsg = boost::make_shared<shape_msgs::Mesh>();
    // geometry_msgs/Point[] 
    mMeshMsg->vertices.resize(vertsSize);
    // shape_msgs/MeshTriangle[] 
    mMeshMsg->triangles.resize(trisSize);

    for(int i = 0; i < vertsSize; i++){
        mMeshMsg->vertices[i].x = mesh.verts[i].x;
        mMeshMsg->vertices[i].y = mesh.verts[i].y;
        mMeshMsg->vertices[i].z = mesh.verts[i].z;
        // std::cout<<mesh.verts[i].x<<std::endl;
    }

    for(int i = 0; i < trisSize; i++){
        mMeshMsg->triangles[i].vertex_indices[0] = mesh.tris[i].a;
        mMeshMsg->triangles[i].vertex_indices[1] = mesh.tris[i].b;
        mMeshMsg->triangles[i].vertex_indices[2] = mesh.tris[i].c;
        // std::cout<<mesh.tris[i].a<<std::endl;
    }
    meshPub.publish(mMeshMsg);
    visual_tools_->publishMesh(T_ws, *mMeshMsg, rviz_visual_tools::ORANGE, 1, "mesh", 1); // rviz_visual_tools::TRANSLUCENT_LIGHT
    // Don't forget to trigger the publisher!
    visual_tools_->trigger();
}

void RosInterface::run() {

  std::thread t_slam([&]() {
    std::cout<<"Start SLAM thread!"<<std::endl;
    cv::Mat img_left, img_right, zedLeftMaskL;
    ros::Time ros_stamp;
    while (ros::ok()) {
      const int64_t timestamp = zed_native->GetStereoFrame(&img_left, &img_right);
      cv::imshow("zed_left", img_left);
      cv::waitKey(1);
      zed_mask_lock.lock();
      zedLeftMaskL = zedLeftMask.clone();
      zed_mask_lock.unlock();
      // my_sys->feed_stereo_frame(img_left, img_right, timestamp);
      my_sys->feed_stereo_frame(img_left, img_right, timestamp, zedLeftMaskL);
      ros_stamp.sec = timestamp / 1000;
      ros_stamp.nsec = (timestamp % 1000) * 1000 * 1000;
      if(mPubZEDImgL.getNumSubscribers()>0)
        publishImage(mZEDImgLBrg, img_left, mPubZEDImgL, "zed", "bgr8" , ros_stamp);
      if(mPubZEDImgR.getNumSubscribers()>0)
        publishImage(mZEDImgRBrg, img_right, mPubZEDImgR, "zed", "bgr8" , ros_stamp);
      ros::spinOnce();
    }
  });

  std::thread t_tsdf([&]() {
    std::cout<<"Start TSDF thread!"<<std::endl;
    cv::Mat img_rgb, img_depth, l515MaskL;
    ros::Time ros_stamp;
    static bool has_started = false;
    while (ros::ok()) {
      // Do not feed data if not tracking yet
      // TODO Possibly should reset mesh when tracking is lost?
      if (my_sys->query_camera_pose(GetSystemTimestamp<std::chrono::milliseconds>()) == SE3<float>::Identity()) {
        continue;
      }
      
      if (!has_started) {
        std::cout << "Starting to build TSDF!" << std::endl;
        has_started = true;
      }

      const int64_t timestamp = l515->GetRGBDFrame(&img_rgb, &img_depth);
      mask_lock.lock();
      l515MaskL = l515Mask.clone();
      mask_lock.unlock();
      // my_sys->feed_rgbd_frame(img_rgb, img_depth, timestamp);
      my_sys->feed_rgbd_frame(img_rgb, img_depth, timestamp,l515MaskL);
      ros_stamp.sec = timestamp / 1000;
      ros_stamp.nsec = (timestamp % 1000) * 1000 * 1000;
      if(mPubL515RGB.getNumSubscribers()>0)
        publishImage(mL515RGBBrg, img_rgb, mPubL515RGB, "l515", "rgb8" , ros_stamp);
      if(mPubL515Depth.getNumSubscribers()>0)
        publishImage(mL515DepthBrg, img_depth, mPubL515Depth, "l515", "mono16", ros_stamp);
      ros::spinOnce();
    }
  });


  std::thread t_reconst([&]() {
      std::cout<<"Start Reconstruction thread!"<<std::endl;
      static unsigned int last_query_time = 0;
      static size_t last_query_amount = 0;
      // static float bbox = 4.0;
      static float x_range[2] = {bbox_x_min, bbox_x_max};
      static float y_range[2] = {bbox_y_min, bbox_y_max};
      static float z_range[2] = {bbox_z_min, bbox_z_max};
      static BoundingCube<float> volumn = {
        x_range[0], x_range[1], y_range[0], y_range[1], z_range[0], z_range[1]};
      static ros::Time stamp;
      ros::Rate rate(4);

    while (ros::ok()) {
    if (!global_mesh)
      {
          float x_off   = transformStamped.transform.translation.x,
                y_off   = transformStamped.transform.translation.y,
                z_off   = transformStamped.transform.translation.z;
          // std::cout << std::setw(15) << "OFFSET: " << std::setprecision(3) << x_off << y_off << z_off << std::endl;
          volumn = {x_off + x_range[0],
                      x_off + x_range[1],
                      y_off + y_range[0],
                      y_off + y_range[1],
                      z_off + z_range[0],
                      z_off + z_range[1]};
      }

      const auto st          = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());
      auto mSemanticReconstr           = my_sys->query_tsdf(volumn);
      const auto end                   = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());
      last_query_time                  = end - st;
      last_query_amount                = mSemanticReconstr.size();
      // std::cout << "Last queried %lu voxels " << last_query_amount << ", took " << last_query_time
      //             << " ms" << std::endl;
      tsdfCb(mSemanticReconstr);
      ros::spinOnce();
      rate.sleep();
    }
  });

  std::thread t_pose([&](){
    std::cout<<"Start Pose thread!"<<std::endl;
    static ros::Time stamp;
    static ros::Rate rate(30);
    while(ros::ok())
    {
      u_int64_t t_query = (int64_t)(GetSystemTimestamp<std::chrono::milliseconds>());
      SE3<float> mSlamPose = my_sys->query_camera_pose(t_query);

      Eigen::Quaternion<float> R = mSlamPose.GetR();
      Eigen::Matrix<float, 3, 1> T = mSlamPose.GetT();
      // std::cout<<"Queried pose at "<<t_query<<std::endl;
      // std::cout << std::setw(5) << "|T: "
      //     << std::setprecision(3)
      //     << std::setw(12) << T.x() << " "
      //     << std::setw(12) << T.y() << " "
      //     << std::setw(12) << T.z() << " ";
      // std::cout << std::setw(5) << "|R: "
      //     << std::setprecision(3)
      //     << std::setw(12) << R.x() << " "
      //     << std::setw(12) << R.y() << " "
      //     << std::setw(12) << R.z() << " "
      //     << std::setw(12) << R.w() << std::endl;

      tf2::Transform tf2_trans;
      tf2::Transform tf2_trans_inv;
      tf2_trans.setRotation(tf2::Quaternion(R.x(), R.y(), R.z(), R.w()));
      tf2_trans.setOrigin(tf2::Vector3(T.x(), T.y(), T.z()));

      stamp.sec  = t_query / 1000; //s
      stamp.nsec = (t_query % 1000) * 1000 * 1000; //ns
      transformStamped.header.stamp = stamp;
      transformStamped.header.frame_id = "slam";
      transformStamped.child_frame_id = "zed";
      tf2_trans_inv = tf2_trans.inverse();
      tf2::Quaternion q = tf2_trans_inv.getRotation();
      transformStamped.transform.rotation.x = q.x();
      transformStamped.transform.rotation.y = q.y();
      transformStamped.transform.rotation.z = q.z();
      transformStamped.transform.rotation.w = q.w();

      tf2::Vector3 t = tf2_trans_inv.getOrigin();
      transformStamped.transform.translation.x = t[0];
      transformStamped.transform.translation.y = t[1];
      transformStamped.transform.translation.z = t[2];
      
      mTfSlam.sendTransform(transformStamped);
      rate.sleep();
    }

  });

  
  if(renderFlag) my_sys->run();
  t_pose.join();
  t_reconst.join();
  t_slam.join();
  t_tsdf.join();
}

void RosInterface::zedMaskCb(const sensor_msgs::ImageConstPtr& msg)
{
  zed_mask_lock.lock();
  zedLeftMask = cv_bridge::toCvShare(msg, "8UC1")->image.clone();
  zed_mask_lock.unlock();
  // try
  // {
  //   cv::imshow("zedMask", zedLeftMask);
  //   cv::waitKey(1);
  // }
  // catch (cv_bridge::Exception& e)
  // {
  //   ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  // }
}

void RosInterface::l515MaskCb(const sensor_msgs::ImageConstPtr& msg)
{
  // image encoding: http://docs.ros.org/en/jade/api/sensor_msgs/html/namespacesensor__msgs_1_1image__encodings.html
  // l515Mask = cv_bridge::toCvShare(msg,  "bgr8")->image;
  // l515Mask.convertTo(l515Mask, CV_8UC1); // depth scale
  mask_lock.lock();
  l515Mask = cv_bridge::toCvShare(msg,  "8UC1")->image.clone();
  mask_lock.unlock();

  // try
  // {
  //   cv::imshow("l515Mask", cv_bridge::toCvShare(msg, "bgr8")->image);
  //   cv::waitKey(30);
  // }
  // catch (cv_bridge::Exception& e)
  // {
  //   ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  // }
}
 