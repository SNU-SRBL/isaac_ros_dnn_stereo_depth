// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_dnn_stereo_decoder/batch_stereo_nodes.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include "isaac_ros_common/cuda_stream.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_dnn_stereo_decoder/filter_disparity.cu.hpp"
#include "isaac_ros_nitros_disparity_image_type/nitros_disparity_image_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace dnn_stereo_depth
{
namespace
{

using TensorList = nvidia::isaac_ros::nitros::NitrosTensorList;
using Tensor = nvidia::isaac_ros::nitros::NitrosTensor;

uint64_t TimestampNs(const TensorList & message)
{
  return static_cast<uint64_t>(message.get_timestamp_sec()) * 1000000000ULL +
         message.get_timestamp_nsec();
}

uint64_t DistanceNs(uint64_t left, uint64_t right)
{
  return left > right ? left - right : right - left;
}

std_msgs::msg::Header HeaderFromTensor(const TensorList & tensor)
{
  std_msgs::msg::Header header;
  header.stamp.sec = static_cast<int32_t>(tensor.get_timestamp_sec());
  header.stamp.nanosec = tensor.get_timestamp_nsec();
  header.frame_id = tensor.get_frame_id();
  return header;
}

std::shared_ptr<Tensor> RequireTensor(
  const TensorList & tensor_list,
  const std::string & name,
  const rclcpp::Logger & logger)
{
  auto tensor = tensor_list.get_tensor_by_name(name);
  if (!tensor) {
    RCLCPP_ERROR(logger, "Tensor '%s' not found", name.c_str());
  }
  return tensor;
}

}  // namespace

TensorBatchPackerNode::TensorBatchPackerNode(const rclcpp::NodeOptions options)
: rclcpp::Node("tensor_batch_packer_node", options),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")},
  left_tensor_topics_{declare_parameter<std::vector<std::string>>(
      "left_tensor_topics", std::vector<std::string>{})},
  right_tensor_topics_{declare_parameter<std::vector<std::string>>(
      "right_tensor_topics", std::vector<std::string>{})},
  left_tensor_name_{declare_parameter<std::string>("left_tensor_name", "left_image")},
  right_tensor_name_{declare_parameter<std::string>("right_tensor_name", "right_image")},
  pair_max_skew_ns_{declare_parameter<int64_t>("pair_max_skew_ms", 1) * 1000000LL},
  batch_max_skew_ns_{declare_parameter<int64_t>("batch_max_skew_ms", 10) * 1000000LL},
  queue_size_{declare_parameter<int>("queue_size", 3)}
{
  if (left_tensor_topics_.empty() || left_tensor_topics_.size() != right_tensor_topics_.size()) {
    throw std::invalid_argument(
            "left_tensor_topics and right_tensor_topics must be non-empty and have equal length");
  }
  if (queue_size_ == 0 || pair_max_skew_ns_ < 0 || batch_max_skew_ns_ < 0) {
    throw std::invalid_argument("queue_size must be positive and skew limits must be non-negative");
  }

  CHECK_CUDA_ERROR(
    nvidia::isaac_ros::common::initNamedCudaStream(stream_, "tensor_batch_packer_node"),
    "Failed to initialize CUDA stream");

  rclcpp::PublisherOptions publisher_options;
  publisher_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  tensor_publisher_ = create_publisher<TensorList>("tensor_pub", output_qos_, publisher_options);

  const size_t camera_count = left_tensor_topics_.size();
  pending_left_.resize(camera_count);
  pending_right_.resize(camera_count);
  pair_queues_.resize(camera_count);
  left_subscribers_.reserve(camera_count);
  right_subscribers_.reserve(camera_count);
  for (size_t index = 0; index < camera_count; ++index) {
    left_subscribers_.push_back(create_subscription<TensorList>(
        left_tensor_topics_[index], input_qos_,
        [this, index](TensorList::ConstSharedPtr tensor) {OnTensor(index, true, std::move(tensor));}));
    right_subscribers_.push_back(create_subscription<TensorList>(
        right_tensor_topics_[index], input_qos_,
        [this, index](TensorList::ConstSharedPtr tensor) {OnTensor(index, false, std::move(tensor));}));
  }
}

TensorBatchPackerNode::~TensorBatchPackerNode()
{
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream_), "Failed to destroy CUDA stream");
}

void TensorBatchPackerNode::OnTensor(
  const size_t camera_index,
  const bool is_left,
  TensorList::ConstSharedPtr tensor)
{
  std::vector<StereoPair> batch;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto & local = is_left ? pending_left_[camera_index] : pending_right_[camera_index];
    auto & opposite = is_left ? pending_right_[camera_index] : pending_left_[camera_index];
    local.push_back(std::move(tensor));
    while (local.size() > static_cast<size_t>(queue_size_)) {
      local.pop_front();
    }

    const auto local_timestamp = TimestampNs(*local.back());
    const auto match = std::min_element(
      opposite.begin(), opposite.end(),
      [local_timestamp](const auto & first, const auto & second) {
        return DistanceNs(TimestampNs(*first), local_timestamp) <
               DistanceNs(TimestampNs(*second), local_timestamp);
      });
    if (match != opposite.end() &&
      DistanceNs(TimestampNs(**match), local_timestamp) <= static_cast<uint64_t>(pair_max_skew_ns_))
    {
      auto paired = *match;
      opposite.erase(match);
      auto current = local.back();
      local.pop_back();
      pair_queues_[camera_index].push_back({
          is_left ? current : paired,
          is_left ? paired : current,
          std::max(TimestampNs(*current), TimestampNs(*paired))});
      while (pair_queues_[camera_index].size() > static_cast<size_t>(queue_size_)) {
        pair_queues_[camera_index].pop_front();
      }
    }
    batch = TakeSynchronizedBatchLocked();
  }
  if (!batch.empty()) {
    PackAndPublish(batch);
  }
}

std::vector<TensorBatchPackerNode::StereoPair> TensorBatchPackerNode::TakeSynchronizedBatchLocked()
{
  while (true) {
    if (std::any_of(
        pair_queues_.begin(), pair_queues_.end(),
        [](const auto & queue) {return queue.empty();}))
    {
      return {};
    }

    uint64_t earliest = std::numeric_limits<uint64_t>::max();
    uint64_t latest = 0;
    size_t earliest_index = 0;
    for (size_t index = 0; index < pair_queues_.size(); ++index) {
      const auto timestamp = pair_queues_[index].front().timestamp_ns;
      if (timestamp < earliest) {
        earliest = timestamp;
        earliest_index = index;
      }
      latest = std::max(latest, timestamp);
    }
    if (latest - earliest > static_cast<uint64_t>(batch_max_skew_ns_)) {
      pair_queues_[earliest_index].pop_front();
      continue;
    }

    std::vector<StereoPair> batch;
    batch.reserve(pair_queues_.size());
    for (auto & queue : pair_queues_) {
      batch.push_back(std::move(queue.front()));
      queue.pop_front();
    }
    return batch;
  }
}

void TensorBatchPackerNode::PackAndPublish(const std::vector<StereoPair> & batch)
{
  const auto first_left = RequireTensor(*batch.front().left, left_tensor_name_, get_logger());
  const auto first_right = RequireTensor(*batch.front().right, right_tensor_name_, get_logger());
  if (!first_left || !first_right ||
    first_left->GetRank() != 4 || first_right->GetRank() != 4 ||
    first_left->GetShape().dims()[0] != 1 || first_right->GetShape().dims()[0] != 1)
  {
    RCLCPP_ERROR(get_logger(), "Batch inputs must be rank-4 tensors with batch dimension 1");
    return;
  }

  const auto pack = [this, &batch](
      const std::string & tensor_name,
      const bool is_left,
      const Tensor & first) -> Tensor
    {
      const auto source_size = first.GetTensorSize();
      const auto source_shape = first.GetShape();
      const auto source_type = first.data_type();
      std::vector<int32_t> batch_shape = source_shape.dims();
      batch_shape[0] = static_cast<int32_t>(batch.size());
      void * data = nullptr;
      CHECK_CUDA_ERROR(
        cudaMallocAsync(&data, source_size * batch.size(), stream_),
        "Failed to allocate batch tensor");

      for (size_t index = 0; index < batch.size(); ++index) {
        const TensorList & source_list = is_left ? *batch[index].left : *batch[index].right;
        const auto source = RequireTensor(source_list, tensor_name, get_logger());
        if (!source || source->GetShape().dims() != source_shape.dims() ||
          source->data_type() != source_type)
        {
          cudaFreeAsync(data, stream_);
          throw std::invalid_argument("All batch tensors must have identical shape and data type");
        }
        CHECK_CUDA_ERROR(
          cudaMemcpyAsync(
            static_cast<uint8_t *>(data) + index * source_size,
            source->GetBuffer(stream_), source_size, cudaMemcpyDeviceToDevice, stream_),
          "Failed to copy input tensor into batch");
      }

      nvidia::isaac_ros::nitros::NitrosTensorBuilder builder;
      return builder.WithName(tensor_name)
             .WithShape(nvidia::isaac_ros::nitros::NitrosTensorShape(batch_shape))
             .WithDataType(source_type)
             .WithData(data)
             .WithReleaseCallback([data, stream = stream_]() {cudaFreeAsync(data, stream);})
             .Build();
    };

  Tensor left;
  Tensor right;
  try {
    left = pack(left_tensor_name_, true, *first_left);
    right = pack(right_tensor_name_, false, *first_right);
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(get_logger(), "Failed to pack tensor batch: %s", exception.what());
    return;
  }

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_), "Failed to synchronize packed batch");
  auto output = nvidia::isaac_ros::nitros::NitrosTensorListBuilder()
    .WithHeader(HeaderFromTensor(*batch.front().left))
    .AddTensor(std::move(left))
    .AddTensor(std::move(right))
    .Build();
  output.set_stream(stream_);
  tensor_publisher_->publish(std::move(output));
}

BatchDNNStereoDecoderNode::BatchDNNStereoDecoderNode(const rclcpp::NodeOptions options)
: rclcpp::Node("batch_dnn_stereo_decoder_node", options),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(*this, "DEFAULT", "output_qos")},
  disparity_tensor_name_{declare_parameter<std::string>("disparity_tensor_name", "disparity")},
  min_disparity_{declare_parameter<double>("min_disparity", 0.0)},
  max_disparity_{declare_parameter<double>("max_disparity", 10000.0)}
{
  const auto camera_info_topics =
    declare_parameter<std::vector<std::string>>(
      "camera_info_topics", std::vector<std::string>{});
  const auto output_topics = declare_parameter<std::vector<std::string>>(
    "output_topics", std::vector<std::string>{});
  if (camera_info_topics.empty() || camera_info_topics.size() != output_topics.size()) {
    throw std::invalid_argument(
            "camera_info_topics and output_topics must be non-empty and have equal length");
  }

  CHECK_CUDA_ERROR(
    nvidia::isaac_ros::common::initNamedCudaStream(stream_, "batch_dnn_stereo_decoder_node"),
    "Failed to initialize CUDA stream");

  rclcpp::PublisherOptions publisher_options;
  publisher_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  camera_infos_.resize(camera_info_topics.size());
  camera_info_subscribers_.reserve(camera_info_topics.size());
  disparity_publishers_.reserve(output_topics.size());
  for (size_t index = 0; index < camera_info_topics.size(); ++index) {
    camera_info_subscribers_.push_back(create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topics[index], input_qos_,
        [this, index](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
          OnCameraInfo(index, std::move(camera_info));
        }));
    disparity_publishers_.push_back(
      create_publisher<nvidia::isaac_ros::nitros::NitrosDisparityImage>(
        output_topics[index], output_qos_, publisher_options));
  }
  tensor_subscriber_ = create_subscription<TensorList>(
    "tensor_sub", input_qos_, [this](TensorList::ConstSharedPtr tensor) {OnTensor(std::move(tensor));});
}

BatchDNNStereoDecoderNode::~BatchDNNStereoDecoderNode()
{
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream_), "Failed to destroy CUDA stream");
}

void BatchDNNStereoDecoderNode::OnCameraInfo(
  const size_t camera_index,
  sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info)
{
  std::lock_guard<std::mutex> lock(mutex_);
  camera_infos_[camera_index] = std::move(camera_info);
}

void BatchDNNStereoDecoderNode::OnTensor(TensorList::ConstSharedPtr tensor_msg)
{
  std::vector<sensor_msgs::msg::CameraInfo::ConstSharedPtr> camera_infos;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    camera_infos = camera_infos_;
  }
  if (std::any_of(
      camera_infos.begin(), camera_infos.end(),
      [](const auto & camera_info) {return camera_info == nullptr;}))
  {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000, "Waiting for all batch camera-info messages");
    return;
  }

  const auto tensor = RequireTensor(*tensor_msg, disparity_tensor_name_, get_logger());
  if (!tensor || tensor->bytes_per_element() != sizeof(float)) {
    RCLCPP_ERROR(get_logger(), "Disparity tensor must be present and float32");
    return;
  }

  const auto rank = tensor->GetRank();
  const auto dimensions = tensor->GetShape().dims();
  if (rank != 3 && rank != 4) {
    RCLCPP_ERROR(get_logger(), "Disparity tensor must have rank 3 [B,H,W] or 4 [B,1,H,W]");
    return;
  }
  if (rank == 4 && dimensions[1] != 1) {
    RCLCPP_ERROR(get_logger(), "Rank-4 disparity tensor must have one channel");
    return;
  }

  const size_t batch_size = static_cast<size_t>(dimensions[0]);
  const uint32_t height = static_cast<uint32_t>(dimensions[rank == 3 ? 1 : 2]);
  const uint32_t width = static_cast<uint32_t>(dimensions[rank == 3 ? 2 : 3]);
  if (batch_size != disparity_publishers_.size()) {
    RCLCPP_ERROR(
      get_logger(), "Disparity batch size %zu does not match configured cameras %zu",
      batch_size, disparity_publishers_.size());
    return;
  }
  const size_t sample_bytes = static_cast<size_t>(height) * width * sizeof(float);
  if (tensor->GetTensorSize() < sample_bytes * batch_size) {
    RCLCPP_ERROR(get_logger(), "Disparity tensor is smaller than its declared batch shape");
    return;
  }

  std::vector<void *> gpu_data(batch_size, nullptr);
  const auto * source = tensor->GetBuffer(stream_);
  for (size_t index = 0; index < batch_size; ++index) {
    const auto & camera_info = *camera_infos[index];
    if (std::abs(camera_info.p[0]) <= std::numeric_limits<double>::epsilon()) {
      RCLCPP_WARN(get_logger(), "Camera %zu has zero focal length; skipping its disparity", index);
      continue;
    }
    CHECK_CUDA_ERROR(
      cudaMallocAsync(&gpu_data[index], sample_bytes, stream_),
      "Failed to allocate disparity output buffer");
    CHECK_CUDA_ERROR(
      cudaMemcpyAsync(
        gpu_data[index], source + index * sample_bytes, sample_bytes,
        cudaMemcpyDeviceToDevice, stream_),
      "Failed to copy batched disparity output");
    CHECK_CUDA_ERROR(
      nvidia::isaac_ros::dnn_stereo_decoder::FilterDisparity(
        static_cast<float *>(gpu_data[index]), width, height,
        static_cast<float>(min_disparity_), static_cast<float>(max_disparity_), stream_),
      "CUDA error after FilterDisparity");
  }
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_), "Failed to synchronize decoded batch");

  for (size_t index = 0; index < batch_size; ++index) {
    if (gpu_data[index] == nullptr) {
      continue;
    }
    std_msgs::msg::Header header = HeaderFromTensor(*tensor_msg);
    header.frame_id = camera_infos[index]->header.frame_id;
    nvidia::isaac_ros::nitros::NitrosDisparityImageBuilder builder;
    auto disparity = builder.WithHeader(header)
      .WithDimensions(height, width)
      .WithGpuData(gpu_data[index])
      .WithDisparityParameters(
        camera_infos[index]->p[0],
        -camera_infos[index]->p[3] / camera_infos[index]->p[0],
        min_disparity_,
        max_disparity_)
      .WithReleaseCallback([data = gpu_data[index], stream = stream_]() {
        cudaFreeAsync(data, stream);
      })
      .Build();
    disparity_publishers_[index]->publish(std::move(disparity));
  }
}

}  // namespace dnn_stereo_depth
}  // namespace isaac_ros
}  // namespace nvidia

RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::dnn_stereo_depth::TensorBatchPackerNode)
RCLCPP_COMPONENTS_REGISTER_NODE(
  nvidia::isaac_ros::dnn_stereo_depth::BatchDNNStereoDecoderNode)
