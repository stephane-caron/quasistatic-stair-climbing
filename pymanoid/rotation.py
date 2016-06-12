#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of pymanoid.
#
# pymanoid is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pymanoid is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# pymanoid. If not, see <http://www.gnu.org/licenses/>.


from math import atan2, asin, cos, sin
from numpy import array, dot


def crossmat(x):
    """Cross-product matrix of a 3D vector"""
    return array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]])


def rpy_from_quat(q):
    roll = atan2(
        2 * q[2] * q[3] + 2 * q[0] * q[1],
        q[3] ** 2 - q[2] ** 2 - q[1] ** 2 + q[0] ** 2)
    pitch = -asin(
        2 * q[1] * q[3] - 2 * q[0] * q[2])
    yaw = atan2(
        2 * q[1] * q[2] + 2 * q[0] * q[3],
        q[1] ** 2 + q[0] ** 2 - q[3] ** 2 - q[2] ** 2)
    return array([roll, pitch, yaw])


def quat_from_rpy(roll, pitch, yaw):
    cr, cp, cy = cos(roll / 2), cos(pitch / 2), cos(yaw / 2)
    sr, sp, sy = sin(roll / 2), sin(pitch / 2), sin(yaw / 2)
    return array([
        cr * cp * cy + sr * sp * sy,
        -cr * sp * sy + cp * cy * sr,
        cr * cy * sp + sr * cp * sy,
        cr * cp * sy - sr * cy * sp])


__quat_to_rot__ = array([[

    # [0, 0]: a^2 + b^2 - c^2 - d^2
    [[+1,  0,  0,  0],
     [.0, +1,  0,  0],
     [.0,  0, -1,  0],
     [.0,  0,  0, -1]],

    # [0, 1]: 2bc - 2ad
    [[.0,  0,  0, -2],
     [.0,  0, +2,  0],
     [.0,  0,  0,  0],
     [.0,  0,  0,  0]],

    # [0, 2]: 2bd + 2ac
    [[.0,  0, +2,  0],
     [.0,  0,  0, +2],
     [.0,  0,  0,  0],
     [.0,  0,  0,  0]]], [

    # [1, 0]: 2bc + 2ad
    [[.0,  0,  0, +2],
     [.0,  0, +2,  0],
     [.0,  0,  0,  0],
     [.0,  0,  0,  0]],

    # [1, 1]: a^2 - b^2 + c^2 - d^2
    [[+1,  0,  0,  0],
     [.0, -1,  0,  0],
     [.0,  0, +1,  0],
     [.0,  0,  0, -1]],

    # [1, 2]: 2cd - 2ab
    [[.0, -2,  0,  0],
     [.0,  0,  0,  0],
     [.0,  0,  0, +2],
     [.0,  0,  0,  0]]], [

    # [2, 0]: 2bd - 2ac
    [[.0,  0, -2,  0],
     [.0,  0,  0, +2],
     [.0,  0,  0,  0],
     [.0,  0,  0,  0]],

    # [2, 1]: 2cd + 2ab
    [[0, +2,  0,  0],
     [0,  0,  0,  0],
     [0,  0,  0, +2],
     [0,  0,  0,  0]],

    # [2, 2]: a^2 - b^2 - c^2 + d^2
    [[+1,  0,  0,  0],
     [.0, -1,  0,  0],
     [.0,  0, -1,  0],
     [.0,  0,  0, +1]]]])

quat_to_rot_tensor = __quat_to_rot__.transpose([2, 0, 1, 3])
# quat_to_rot_tensor.shape == (4, 3, 3, 4)


def rotation_matrix_from_quat(quat):
    return dot(quat, dot(quat_to_rot_tensor, quat))
