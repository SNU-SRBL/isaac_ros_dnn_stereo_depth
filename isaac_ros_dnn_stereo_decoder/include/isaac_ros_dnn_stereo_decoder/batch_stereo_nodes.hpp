// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#ifndef ISAAC_ROS_DNN_STEREO_DECODER__BATCH_STEREO_NODES_HPP_
#define ISAAC_ROS_DNN_STEREO_DECODER__BATCH_STEREO_NODES_HPP_

#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_depth
{

class TensorBatchPackerNode : public rclcpp::Node
{
public:
  explicit TensorBatchPackerNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~TensorBatchPackerNode() override;

private:
  using TensorList = nvidia::isaac_ros::nitros::NitrosTensorList;

  struct StereoPair
  {
    TensorList::ConstSharedPtr left;
    TensorList::ConstSharedPtr right;
    uint64_t timestamp_ns;
  };

  void OnTensor(size_t camera_index, bool is_left, TensorList::ConstSharedPtr tensor);
  std::vector<StereoPair> TakeSynchronizedBatchLocked();
  void PackAndPublish(const std::vector<StereoPair> & batch);

  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;
  std::vector<std::string> left_tensor_topics_;
  std::vector<std::string> right_tensor_topics_;
  std::string left_tensor_name_;
  std::string right_tensor_name_;
  int64_t pair_max_skew_ns_;
  int64_t batch_max_skew_ns_;
  int64_t queue_size_;
  cudaStream_t stream_{nullptr};
  std::vector<rclcpp::Subscription<TensorList>::SharedPtr> left_subscribers_;
  std::vector<rclcpp::Subscription<TensorList>::SharedPtr> right_subscribers_;
  rclcpp::Publisher<TensorList>::SharedPtr tensor_publisher_;
  std::vector<std::deque<TensorList::ConstSharedPtr>> pending_left_;
  std::vector<std::deque<TensorList::ConstSharedPtr>> pending_right_;
  std::vector<std::deque<StereoPair>> pair_queues_;
  std::mutex mutex_;
};

class BatchDNNStereoDecoderNode : public rclcpp::Node
{
public:
  explicit BatchDNNStereoDecoderNode(const rclcpp::NodeOptions options = rclcpp::NodeOptions());
  ~BatchDNNStereoDecoderNode() override;

private:
  using TensorList = nvidia::isaac_ros::nitros::NitrosTensorList;

  void OnCameraInfo(size_t camera_index, sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info);
  void OnTensor(TensorList::ConstSharedPtr tensor_msg);

  rclcpp::QoS input_qos_;
  rclcpp::QoS output_qos_;
  std::string disparity_tensor_name_;
  double min_disparity_;
  double max_disparity_;
  cudaStream_t stream_{nullptr};
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr> camera_info_subscribers_;
  rclcpp::Subscription<TensorList>::SharedPtr tensor_subscriber_;
  std::vector<sensor_msgs::msg::CameraInfo::ConstSharedPtr> camera_infos_;
  std::vector<rclcpp::Publisher<nvidia::isaac_ros::nitros::NitrosDisparityImage>::SharedPtr>
    disparity_publishers_;
  std::mutex mutex_;
};

}  // namespace dnn_stereo_depth
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // ISAAC_ROS_DNN_STEREO_DECODER__BATCH_STEREO_NODES_HPP_
