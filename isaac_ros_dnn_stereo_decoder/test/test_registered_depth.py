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
from isaac_ros_dnn_stereo_decoder import registered_depth
from isaac_ros_dnn_stereo_decoder.registered_depth import registered_depth_mm
import numpy as np
import pytest
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


def test_registered_depth_splats_fractional_projection_without_blank_rows():
    source = CameraInfo()
    source.width = source.height = 4
    source.k = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    color = CameraInfo()
    color.width = color.height = 4
    color.k = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    message = DisparityImage()
    message.f = 1.0
    message.t = 0.1
    message.image = Image()
    message.image.height = message.image.width = 4
    message.image.encoding = "32FC1"
    message.image.step = 16
    message.image.data = np.full((4, 4), 0.1, dtype=np.float32).tobytes()
    transform = _identity_transform()
    transform.translation.y = 0.5

    depth = registered_depth_mm(message, source, color, transform)
    assert depth.dtype == np.uint16
    assert np.all(depth > 0)


def test_registered_depth_splat_matches_lexsort_for_edges_cutoff_and_z16_max():
    width, height = 3, 2
    u0 = np.array([0, 0, -1, 2, 1], dtype=np.int64)
    v0 = np.array([0, 0, 0, 1, 0], dtype=np.int64)
    du = np.array([0.0, 0.0, 0.5, 0.5, 5e-7])
    dv = np.array([0.0, 0.0, 0.5, 0.5, 0.0])
    z_color = np.array([2.0, 1.0, 65.535, 65.535, 3.0])

    pixels, depths = [], []
    for dx, dy, weight in (
        (0, 0, (1.0 - du) * (1.0 - dv)),
        (1, 0, du * (1.0 - dv)),
        (0, 1, (1.0 - du) * dv),
        (1, 1, du * dv),
    ):
        x, y = u0 + dx, v0 + dy
        keep = (
            (weight > 1e-6)
            & (x >= 0)
            & (x < width)
            & (y >= 0)
            & (y < height)
        )
        pixels.append(y[keep] * width + x[keep])
        depths.append(z_color[keep])
    flat = np.concatenate(pixels)
    z_mm = np.rint(np.concatenate(depths) * 1000.0).clip(1, 65535).astype(np.uint16)
    order = np.lexsort((z_mm, flat))
    flat, z_mm = flat[order], z_mm[order]
    first = np.empty(flat.size, dtype=bool)
    first[0] = True
    first[1:] = flat[1:] != flat[:-1]
    expected = np.zeros((height, width), dtype=np.uint16)
    expected.reshape(-1)[flat[first]] = z_mm[first]

    actual = np.zeros_like(expected)
    registered_depth._splat_registered_depth(
        actual.reshape(-1), u0, v0, du, dv, z_color, width
    )

    assert np.array_equal(actual, expected)
    assert actual[0, 0] == 1000
    assert actual[0, 1] == 3000  # The sub-threshold right-hand splat is omitted.
    assert actual[1, 0] == 65535
    assert actual[1, 2] == 65535


def test_main_does_not_shutdown_an_already_stopped_context(monkeypatch):
    class FakeNode:
        def destroy_node(self):
            pass

    def interrupt_spin(_node):
        raise KeyboardInterrupt

    monkeypatch.setattr(registered_depth.rclpy, "init", lambda args: None)
    monkeypatch.setattr(registered_depth, "RegisteredDepthNode", FakeNode)
    monkeypatch.setattr(registered_depth.rclpy, "spin", interrupt_spin)
    monkeypatch.setattr(registered_depth.rclpy, "ok", lambda: False)
    monkeypatch.setattr(
        registered_depth.rclpy,
        "shutdown",
        lambda: pytest.fail("shutdown must not run twice"),
    )

    with pytest.raises(KeyboardInterrupt):
        registered_depth.main()
