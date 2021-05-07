#include "utils/offline_data_provider/rosbag_reader.h"

#include <assert.h>
#include <spdlog/spdlog.h>

#include <iostream>
#include <memory>

rosbag_reader::rosbag_reader(const string& rosbag_path) {
  rosbag::Bag my_bag_(rosbag_path, rosbag::bagmode::Read);

  process_metadata(my_bag_);
  process_streamdata(my_bag_);
  process_stereodata(my_bag_);
}

CameraIntrinsics<float> rosbag_reader::get_camera_intrinsics() {
  std::istringstream str_stream(intrinsics_str_);
  float buff[4];

  str_stream >> buff[0] >> buff[1] >> buff[2] >> buff[3];

  return CameraIntrinsics<float>(buff[0], buff[1], buff[2], buff[3]);
}

SE3<float> rosbag_reader::get_camera_extrinsics() {
  if (extrinsics_str_ == "") return SE3<float>::Identity();

  std::istringstream str_stream(extrinsics_str_);
  float buff[16];

  /* read 16 float numbers from the string */
  str_stream >> buff[0] >> buff[1] >> buff[2] >> buff[3] >> buff[4] >> buff[5] >> buff[6] >>
      buff[7] >> buff[8] >> buff[9] >> buff[10] >> buff[11] >> buff[12] >> buff[13] >> buff[14] >>
      buff[15];
  const Eigen::Matrix<float, 4, 4> tmp =
      Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(buff);
  return SE3<float>(tmp);
}

float rosbag_reader::get_depth_map_factor() { return 1000; }

void rosbag_reader::get_depth_frame_by_id(cv::Mat* depth_img, int frame_idx) {
  if ((frame_idx < 0) || (frame_idx >= size_)) {
    spdlog::error("Invalid frame idx {} supplied!", frame_idx);
  }
  cv::resize(depth_img_vec_[frame_idx], *depth_img, cv::Size(width, height));
}

void rosbag_reader::get_color_frame_by_id(cv::Mat* rgb_img, int frame_idx) {
  if ((frame_idx < 0) || (frame_idx >= size_)) {
    spdlog::error("Invalid frame idx {} supplied!", frame_idx);
  }
  cv::resize(rgb_img_vec_[frame_idx], *rgb_img, cv::Size(width, height));
}

SE3<float> rosbag_reader::get_camera_pose_by_id(int frame_idx) {
  if ((frame_idx < 0) || (frame_idx >= size_)) {
    spdlog::error("Invalid frame idx {} supplied!", frame_idx);
  }
  return pose_vec_[frame_idx];
}

int rosbag_reader::get_size() { return size_; }

int rosbag_reader::get_width() { return width; }

int rosbag_reader::get_height() { return height; }

void rosbag_reader::process_metadata(const rosbag::Bag& my_bag_) {
  std::vector<std::string> meta_topics;
  meta_topics.push_back(depth_factor_topic);
  meta_topics.push_back(extrinsics_topic);
  meta_topics.push_back(intrinsics_topic);

  rosbag::View metadata_view(my_bag_, rosbag::TopicQuery(meta_topics));

  /* Read Meta Data */
  for (const auto& m : metadata_view) {
    if (m.getTopic() == depth_factor_topic) {
      std_msgs::String::ConstPtr meta_ptr = m.instantiate<std_msgs::String>();
      depth_factor_str_ = meta_ptr->data;
    }
    if (m.getTopic() == extrinsics_topic) {
      std_msgs::String::ConstPtr meta_ptr = m.instantiate<std_msgs::String>();
      extrinsics_str_ = meta_ptr->data;
    }
    if (m.getTopic() == intrinsics_topic) {
      std_msgs::String::ConstPtr meta_ptr = m.instantiate<std_msgs::String>();
      intrinsics_str_ = meta_ptr->data;
    }
  }
}

void rosbag_reader::process_streamdata(const rosbag::Bag& my_bag_) {
  pose_manager my_posemanager;

  std::vector<std::string> stream_topics;
  stream_topics.push_back(depth_img_topic);
  stream_topics.push_back(rgb_img_topic);

  rosbag::View stream_view(my_bag_, rosbag::TopicQuery(stream_topics));

  /* Data are read w.r.t. timestamps. Different types of data may come in every loop */
  for (const auto& m : stream_view) {
    if (m.getTopic() == depth_img_topic) {
      sensor_msgs::Image::ConstPtr sensor_ptr = m.instantiate<sensor_msgs::Image>();
      if (sensor_ptr != nullptr) {
        auto depth_img_ptr =
            cv_bridge::toCvCopy(sensor_ptr, sensor_msgs::image_encodings::TYPE_16UC1);
        cv::Mat depth_img = depth_img_ptr->image;

        /* in millisecond */
        int64_t timestamp = (int64_t)(depth_img_ptr->header.stamp.toNSec());
        depth_img_vec_.push_back(depth_img);
        depth_ts_vec_.push_back(timestamp);
      }
    }
    if (m.getTopic() == rgb_img_topic) {
      sensor_msgs::Image::ConstPtr sensor_ptr = m.instantiate<sensor_msgs::Image>();
      if (sensor_ptr != nullptr) {
        auto rgb_img_ptr = cv_bridge::toCvCopy(sensor_ptr, sensor_msgs::image_encodings::TYPE_8UC3);
        cv::Mat rgb_img = rgb_img_ptr->image;

        /* by default, ROSBag save RGB images; while OpenCV reads BGR. */
        cv::cvtColor(rgb_img, rgb_img, cv::COLOR_RGB2BGR);

        /* in millisecond */
        int64_t timestamp = (int64_t)(rgb_img_ptr->header.stamp.toNSec());
        rgb_img_vec_.push_back(rgb_img);
        rgb_ts_vec_.push_back(timestamp);
      }
    }
    if (m.getTopic() == pose_topic) {
      geometry_msgs::PoseStamped::ConstPtr camera_pose =
          m.instantiate<geometry_msgs::PoseStamped>();
      if (camera_pose != nullptr) {
        Eigen::Quaternion<float> q;
        q.x() = camera_pose->pose.orientation.x;
        q.y() = camera_pose->pose.orientation.y;
        q.z() = camera_pose->pose.orientation.z;
        q.w() = camera_pose->pose.orientation.w;
        Eigen::Matrix<float, 3, 1> trans;
        trans(0, 0) = camera_pose->pose.position.x;
        trans(1, 0) = camera_pose->pose.position.y;
        trans(2, 0) = camera_pose->pose.position.z;

        SE3<float> cur_pose(q, trans);

        int64_t timestamp = (int64_t)(camera_pose->header.stamp.toNSec());

        my_posemanager.register_valid_pose(timestamp, cur_pose.Inverse());
      }
    }
  }

  /* Populate size */
  size_ = (int)std::min(rgb_ts_vec_.size(), depth_ts_vec_.size());
}

void rosbag_reader::process_stereodata(const rosbag::Bag& my_bag_) {
  pose_manager my_posemanager;

  std::vector<std::string> stereo_topics;
  stereo_topics.push_back(left_img_topic);
  stereo_topics.push_back(right_img_topic);

  rosbag::View stereo_view(my_bag_, rosbag::TopicQuery(stereo_topics));

  std::vector<int64_t> left_ts_vec_;
  std::vector<int64_t> right_ts_vec_;
  std::vector<cv::Mat> left_img_vec_;
  std::vector<cv::Mat> right_img_vec_;

  /* Data are read w.r.t. timestamps. Different types of data may come in every loop */
  for (const auto& m : stereo_view) {
    if (m.getTopic() == left_img_topic) {
      sensor_msgs::Image::ConstPtr sensor_ptr = m.instantiate<sensor_msgs::Image>();
      if (sensor_ptr != nullptr) {
        auto rgb_img_ptr = cv_bridge::toCvCopy(sensor_ptr, sensor_msgs::image_encodings::TYPE_8UC4);
        cv::Mat rgb_img = rgb_img_ptr->image;

        /* by default, ROSBag save RGB images; while OpenCV reads BGR. */
        cv::cvtColor(rgb_img, rgb_img, cv::COLOR_RGBA2BGR);

        /* in millisecond */
        int64_t timestamp = (int64_t)(rgb_img_ptr->header.stamp.toNSec());
        left_img_vec_.push_back(rgb_img);
        left_ts_vec_.push_back(timestamp);
      }
    }
    if (m.getTopic() == right_img_topic) {
      sensor_msgs::Image::ConstPtr sensor_ptr = m.instantiate<sensor_msgs::Image>();
      if (sensor_ptr != nullptr) {
        auto rgb_img_ptr = cv_bridge::toCvCopy(sensor_ptr, sensor_msgs::image_encodings::TYPE_8UC4);
        cv::Mat rgb_img = rgb_img_ptr->image;

        /* by default, ROSBag save RGB images; while OpenCV reads BGR. */
        cv::cvtColor(rgb_img, rgb_img, cv::COLOR_RGBA2BGR);

        /* in millisecond */
        int64_t timestamp = (int64_t)(rgb_img_ptr->header.stamp.toNSec());
        right_img_vec_.push_back(rgb_img);
        right_ts_vec_.push_back(timestamp);
      }
    }
  }

  int valid_stereo_cnt_ = (int)std::min(left_ts_vec_.size(), right_ts_vec_.size());

  /* Initialize SLAM module */
  YAML::Node yaml_node = YAML::LoadFile(camera_config_path);
  std::shared_ptr<openvslam::config> cfg = GetAndSetConfig(camera_config_path);
  auto my_slam_sys = std::make_shared<SLAMSystem>(cfg, vocab_path);
  my_slam_sys->startup();

  /* Send these images to OpenVSLAM */
  std::vector<unsigned int> frame_idx_list;

  for (size_t i = 0; i < valid_stereo_cnt_; ++i) {
    spdlog::info("Feeding image {}/{}", i, valid_stereo_cnt_);
    int64_t cur_timestamp = std::min(left_ts_vec_[i], right_ts_vec_[i]);
    unsigned int frame_id = my_slam_sys->FeedStereoImages(left_img_vec_[i], right_img_vec_[i], cur_timestamp / 1e9);
    frame_idx_list.push_back(frame_id);
  }

  /* Extract computed/loop-closed poses from SLAM module */
  std::vector<Eigen::Matrix4d> extracted_poses = my_slam_sys->get_saved_trajectory(frame_idx_list);
  my_slam_sys->SaveMatchedTrajectory("/tmp/icu_trajectory.txt", frame_idx_list);
  assert (extracted_poses.size() == (size_t)valid_stereo_cnt_);

  /* Register SLAM poses */
  for (size_t i = 0; i < valid_stereo_cnt_; ++i) {
    int64_t cur_timestamp = std::min(left_ts_vec_[i], right_ts_vec_[i]);
    my_posemanager.register_valid_pose(cur_timestamp, extracted_poses[i]);
  }

  /* Fill pose_vec_  */
  for (size_t i = 0; i < size_; ++i) {
    int64_t cur_timestamp = depth_ts_vec_[i];
    SE3<float> cur_pose = my_posemanager.query_pose(cur_timestamp);
    pose_vec_.push_back(cur_pose);
  }

  my_slam_sys->shutdown();
}
