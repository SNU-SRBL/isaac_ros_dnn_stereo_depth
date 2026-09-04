# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Register calibrated stereo disparity as a color-aligned metric depth image."""

from collections import deque
from copy import deepcopy

from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from stereo_msgs.msg import DisparityImage


def _stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _rotation_matrix(transform):
    q = transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = x * x + y * y + z * z + w * w
    if norm <= 0.0:
        raise ValueError('extrinsics rotation must be non-zero')
    x, y, z, w = x / norm**0.5, y / norm**0.5, z / norm**0.5, w / norm**0.5
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def registered_depth_mm(disparity, left_info, color_info, ir_to_color, max_depth_m=65.535):
    """
    Project a disparity image into the color camera and return Z16 millimetres.

    ``ir_to_color`` maps points expressed in the left stereo camera into the
    color camera. The nearest projected point wins each color pixel, matching
    the occlusion semantics expected from an aligned depth image.
    """
    if disparity.image.encoding != '32FC1':
        raise ValueError(
            f'Expected disparity encoding 32FC1, got {disparity.image.encoding!r}'
        )
    height, width = disparity.image.height, disparity.image.width
    if height <= 0 or width <= 0 or disparity.image.step < width * 4:
        raise ValueError('invalid disparity image dimensions')
    values = np.frombuffer(disparity.image.data, dtype=np.float32)
    if values.size < height * (disparity.image.step // 4):
        raise ValueError('disparity image payload is shorter than its stride')
    values = values.reshape(height, disparity.image.step // 4)[:, :width]

    f, baseline = float(disparity.f), float(disparity.t)
    if f <= 0.0 or baseline <= 0.0:
        raise ValueError('disparity focal length and baseline must be positive')
    fx, fy, cx, cy = map(
        float, (left_info.k[0], left_info.k[4], left_info.k[2], left_info.k[5])
    )
    cfx, cfy, ccx, ccy = map(
        float, (color_info.k[0], color_info.k[4], color_info.k[2], color_info.k[5])
    )
    if min(fx, fy, cfx, cfy) <= 0.0:
        raise ValueError('camera-info focal lengths must be positive')
    if color_info.width <= 0 or color_info.height <= 0:
        raise ValueError('invalid color camera-info dimensions')

    depth = np.divide(
        f * baseline, values, out=np.zeros_like(values), where=values > 0.0
    )
    valid = np.isfinite(depth) & (values > 0.0) & (depth > 0.0) & (depth <= max_depth_m)
    rows, cols = np.nonzero(valid)
    output = np.zeros((color_info.height, color_info.width), dtype=np.uint16)
    if not len(rows):
        return output

    z = depth[rows, cols]
    points = np.stack(((cols - cx) * z / fx, (rows - cy) * z / fy, z), axis=1)
    rotated = points @ _rotation_matrix(ir_to_color).T
    translation = ir_to_color.translation
    rotated += np.array([translation.x, translation.y, translation.z], dtype=np.float32)
    z_color = rotated[:, 2]
    valid = np.isfinite(z_color) & (z_color > 0.0) & (z_color <= max_depth_m)
    u = np.rint(cfx * rotated[:, 0] / z_color + ccx).astype(np.int64)
    v = np.rint(cfy * rotated[:, 1] / z_color + ccy).astype(np.int64)
    valid &= (u >= 0) & (u < color_info.width) & (v >= 0) & (v < color_info.height)
    if not np.any(valid):
        return output

    flat = v[valid] * color_info.width + u[valid]
    z_mm = np.rint(z_color[valid] * 1000.0).clip(1, 65535).astype(np.uint16)
    # Sort by pixel then depth: the first entry is the visible surface.
    order = np.lexsort((z_mm, flat))
    flat, z_mm = flat[order], z_mm[order]
    first = np.empty(flat.size, dtype=bool)
    first[0] = True
    first[1:] = flat[1:] != flat[:-1]
    output.reshape(-1)[flat[first]] = z_mm[first]
    return output


class RegisteredDepthNode(Node):
    """Publish a RealSense-compatible aligned Z16 depth image from disparity."""

    def __init__(self):
        super().__init__('registered_depth')
        self._max_sync_ns = int(
            self.declare_parameter('max_sync_ms', 75).value * 1_000_000
        )
        self._max_depth_m = float(self.declare_parameter('max_depth_m', 65.535).value)
        # B2 inference can lag the source color frame by over 400 ms at 30 Hz.
        self._max_queue_size = int(self.declare_parameter('queue_size', 16).value)
        self._left_info = None
        self._color_info = None
        self._ir_to_color = None
        self._pending_disparities = deque()
        self._color_images = deque()
        camera_info_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                     durability=DurabilityPolicy.VOLATILE)
        transient_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                   durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._depth_pub = self.create_publisher(
            Image, self.declare_parameter('output_depth_topic').value, qos_profile_sensor_data)
        self._info_pub = self.create_publisher(
            CameraInfo, self.declare_parameter('output_camera_info_topic').value, camera_info_qos)
        self.create_subscription(
            DisparityImage, self.declare_parameter('disparity_topic').value,
            self._on_disparity, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, self.declare_parameter('left_camera_info_topic').value,
            self._on_left_info, camera_info_qos)
        self.create_subscription(
            Image, self.declare_parameter('color_image_topic').value,
            self._on_color_image, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, self.declare_parameter('color_camera_info_topic').value,
            self._on_color_info, camera_info_qos)
        self.create_subscription(
            TransformStamped, self.declare_parameter('extrinsics_topic').value,
            self._on_extrinsics, transient_qos)

    def _on_left_info(self, message):
        self._left_info = message

    def _on_color_info(self, message):
        self._color_info = message

    def _on_extrinsics(self, message):
        self._ir_to_color = message.transform

    def _on_color_image(self, message):
        self._color_images.append(message)
        while len(self._color_images) > self._max_queue_size:
            self._color_images.popleft()
        self._publish_ready()

    def _on_disparity(self, message):
        self._pending_disparities.append(message)
        while len(self._pending_disparities) > self._max_queue_size:
            self._pending_disparities.popleft()
        self._publish_ready()

    def _publish_ready(self):
        if not all((self._left_info, self._color_info, self._ir_to_color, self._color_images)):
            return
        retained = deque()
        while self._pending_disparities:
            disparity = self._pending_disparities.popleft()
            stamp = _stamp_ns(disparity.header.stamp)
            color = min(
                self._color_images,
                key=lambda item: abs(_stamp_ns(item.header.stamp) - stamp),
            )
            if abs(_stamp_ns(color.header.stamp) - stamp) > self._max_sync_ns:
                retained.append(disparity)
                continue
            try:
                depth_mm = registered_depth_mm(
                    disparity, self._left_info, self._color_info, self._ir_to_color,
                    self._max_depth_m,
                )
            except ValueError as error:
                self.get_logger().error(str(error))
                continue
            output = Image()
            output.header = color.header
            output.height, output.width = depth_mm.shape
            output.encoding = '16UC1'
            output.is_bigendian = 0
            output.step = output.width * 2
            output.data = depth_mm.tobytes()
            self._depth_pub.publish(output)
            info = deepcopy(self._color_info)
            info.header = color.header
            self._info_pub.publish(info)
        self._pending_disparities = retained


def main(args=None):
    rclpy.init(args=args)
    node = RegisteredDepthNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
