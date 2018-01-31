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

import openravepy

from numpy import array, dot, eye
from openravepy import matrixFromPose


FOOT_X = 0.120   # half-length
FOOT_Y = 0.065   # half-width
FOOT_Z = 0.027   # half-height


class Box(object):

    def __init__(self, env, name, dims, pose, color, transparency=0.):
        self.T = eye(4)
        self.env = env
        self.X = dims[0]
        self.Y = dims[1]
        self.Z = dims[2]
        self.name = name
        self.body = openravepy.RaveCreateKinBody(env, '')
        self.body.SetName(name)
        self.body.InitFromBoxes(array([
            array([0., 0., 0., self.X, self.Y, self.Z])]))
        self.set_color(color)
        env.Add(self.body)
        if pose is not None:
            self.set_transform_pose(pose)
        if transparency > 0:
            self.set_transparency(transparency)

    def set_color(self, color):
        r, g, b = \
            [.9, .5, .5] if color == 'r' else \
            [.2, 1., .2] if color == 'g' else \
            [.2, .2, 1.]  # if color == 'b'
        for link in self.body.GetLinks():
            for geom in link.GetGeometries():
                geom.SetAmbientColor([r, g, b])
                geom.SetDiffuseColor([r, g, b])

    def set_visibility(self, visible):
        self.body.SetVisible(visible)

    def set_transparency(self, transparency):
        for link in self.body.GetLinks():
            for geom in link.GetGeometries():
                geom.SetTransparency(transparency)

    def __del__(self):
        self.env.Remove(self.body)

    def set_transform_pose(self, pose):
        self.pose = pose
        self.T = matrixFromPose(pose)
        self.R = self.T[:3, :3]
        self.p = self.T[:3, 3]
        self.body.SetTransform(self.T)

    @property
    def corners(self):
        assert self.is_foot, "Trying to get corners of point contact"
        return [
            dot(self.T, array([+self.X, +self.Y, -self.Z, 1.]))[:3],
            dot(self.T, array([+self.X, -self.Y, -self.Z, 1.]))[:3],
            dot(self.T, array([-self.X, +self.Y, -self.Z, 1.]))[:3],
            dot(self.T, array([-self.X, -self.Y, -self.Z, 1.]))[:3]]

    def collides_with(self, other_pseudo):
        return self.env.CheckCollision(self.body, other_pseudo.body)


class PseudoFoot(Box):

    def __init__(self, env, name, pose=None, color='g', transparency=0.2):
        super(PseudoFoot, self).__init__(
            env, name, [FOOT_X, FOOT_Y, FOOT_Z], pose, color, transparency)

    @property
    def target(self):
        return self.pose[4:] + array([0., 0., 0.16])
