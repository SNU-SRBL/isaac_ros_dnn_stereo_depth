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

from geometry_msgs.msg import Transform
from isaac_ros_dnn_stereo_decoder.registered_depth import registered_depth_mm
import numpy as np
from sensor_msgs.msg import CameraInfo, Image
from stereo_msgs.msg import DisparityImage


def _camera_info(width=2, height=2):
    info = CameraInfo()
    info.width = width
    info.height = height
    info.k = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    return info


def _disparity(values):
    message = DisparityImage()
    message.f = 1.0
    message.t = 0.1
    message.image = Image()
    message.image.height = 2
    message.image.width = 2
    message.image.encoding = '32FC1'
    message.image.step = 8
    message.image.data = np.asarray(values, dtype=np.float32).reshape(2, 2).tobytes()
    return message


def _identity_transform():
    transform = Transform()
    transform.rotation.w = 1.0
    return transform


def test_registered_depth_converts_disparity_to_z16_millimetres():
    depth = registered_depth_mm(
        _disparity([[0.1, 0.1], [0.1, 0.0]]),
        _camera_info(),
        _camera_info(),
        _identity_transform(),
    )
    assert depth.dtype == np.uint16
    assert depth.tolist() == [[1000, 1000], [1000, 0]]


def test_registered_depth_uses_nearest_point_for_overlapping_projection():
    color = _camera_info(width=1, height=1)
    color.k = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0]
    depth = registered_depth_mm(
        _disparity([[0.1, 0.2], [0.1, 0.2]]),
        _camera_info(),
        color,
        _identity_transform(),
    )
    assert depth.tolist() == [[500]]
