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

import cdd
import matplotlib
import pylab
import pymanoid.bodies as bodies

from numpy import array, dot, hstack, ones, zeros
from pylab import norm
from pymanoid.inverse_kinematics import VelocityConstraint
from pymanoid.inverse_kinematics import VelocityTracker
from pymanoid.rave import display_box
from pymanoid.trajectory import Trajectory, LinearChunk
from scipy.linalg import block_diag
from scipy.spatial import ConvexHull


kin_dt = 5e-3
w_com = 1.
w_link = 2e-1
polygon_colors = ['b', 'g']
polygon_color_index = 0
all_plots = False


def plot_polygon(poly, alpha=.4):
    global polygon_color_index
    color = polygon_colors[polygon_color_index]
    polygon_color_index = (polygon_color_index + 1) % len(polygon_colors)
    if type(poly) is list:
        poly = array(poly)
    pylab.ion()
    ax = pylab.gca()
    hull = ConvexHull(poly)
    poly = poly[hull.vertices, :]
    xmin1, xmax1, ymin1, ymax1 = pylab.axis()
    xmin2, ymin2 = poly.min(axis=0)
    xmax2, ymax2 = poly.max(axis=0)
    pylab.axis((min(xmin1, xmin2), max(xmax1, xmax2),
                min(ymin1, ymin2), max(ymax1, ymax2)))
    patch = matplotlib.patches.Polygon(poly, alpha=alpha, color=color)
    ax.add_patch(patch)


class TrajectorySketch(object):

    def __init__(self, robot, q_init, q_upper_ref=None):
        self.chunks = []
        self.cur_q = q_init
        self.contacting_links = []
        self.robot = robot
        self.q_upper_ref = q_upper_ref
        if q_upper_ref is None:
            self.q_upper_ref = self.robot.q_halfsit

    @property
    def cur_com(self):
        return self.robot.compute_com(self.cur_q)

    def contact_link(self, link):
        self.contacting_links.append(link)

    def free_link(self, link):
        self.contacting_links.remove(link)

    def get_trajectory(self):
        return Trajectory(self.chunks)

    def fix_link(self, tracker, link):
        tracker.add_constraint(VelocityConstraint(
            vel_fun=lambda t: zeros(6),
            jacobian_fun=lambda q: self.robot.compute_link_jacobian(link, q)))

    def attract_dof(self, tracker, dof, ref_value):
        def vel_fun(t):
            q = self.robot.get_dof_values()
            return ref_value - q[dof]

        J = zeros(len(self.cur_q))
        J[dof] = 1.
        tracker.add_objective(VelocityConstraint(
            vel_fun, lambda q: J, gain=1.), weight=1e-3)

    def init_tracker(self, duration):
        tracker = VelocityTracker(
            self.robot, self.cur_q, duration, kin_dt)
        for link in self.contacting_links:
            self.fix_link(tracker, link)
        for dof in set(self.robot.upper_dofs) | set(self.robot.base_rot_dofs):
            self.attract_dof(tracker, dof, ref_value=self.q_upper_ref[dof])
        return tracker

    def check_com_positions(self, com_positions):
        X = bodies.FOOT_X
        Y = bodies.FOOT_Y
        m = self.robot.mass
        g = 9.81
        mu = 0.7
        CWC = array([
            # fx  fy              fz  taux tauy tauz
            [-1,   0,            -mu,    0,   0,   0],
            [+1,   0,            -mu,    0,   0,   0],
            [0,   -1,            -mu,    0,   0,   0],
            [0,   +1,            -mu,    0,   0,   0],
            [0,    0,             -Y,   -1,   0,   0],
            [0,    0,             -Y,   +1,   0,   0],
            [0,    0,             -X,    0,  -1,   0],
            [0,    0,             -X,    0,  +1,   0],
            [-Y,  -X,  -(X + Y) * mu,  +mu,  +mu,  -1],
            [-Y,  +X,  -(X + Y) * mu,  +mu,  -mu,  -1],
            [+Y,  -X,  -(X + Y) * mu,  -mu,  +mu,  -1],
            [+Y,  +X,  -(X + Y) * mu,  -mu,  -mu,  -1],
            [+Y,  +X,  -(X + Y) * mu,  +mu,  +mu,  +1],
            [+Y,  -X,  -(X + Y) * mu,  +mu,  -mu,  +1],
            [-Y,  +X,  -(X + Y) * mu,  -mu,  +mu,  +1],
            [-Y,  -X,  -(X + Y) * mu,  -mu,  -mu,  +1]])
        nb_contacts = len(self.contacting_links)
        C = zeros((4, 6 * nb_contacts))
        d = array([0, 0, -m * g, 0])
        # [pGx, pGy] = D * w_all
        D = zeros((2, 6 * nb_contacts))
        for i, link in enumerate(self.contacting_links):
            # check orientation assumption
            pose = link.GetTransformPose()
            assert norm(pose[:4] - array([1., 0., 0., 0.])) < 5e-2, \
                str(float(norm(pose[:4] - array([1., 0., 0., 0.]))))

            x, y, z = link.GetTransformPose()[4:]
            Ci = array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [-y, x, 0, 0, 0, 1]])
            Di = 1. / (m * g) * array([
                [-z, 0, x, 0, -1, 0],
                [0, -z, y, 1,  0, 0]])
            C[0:4, (6 * i):(6 * (i + 1))] = +Ci
            D[:, (6 * i):(6 * (i + 1))] = Di

        CWC_all = block_diag(*([CWC] * nb_contacts))
        _zeros = zeros((CWC_all.shape[0], 1))
        # A * w_all + b >= 0
        # input to cdd.Matrix is [b, A]
        F = cdd.Matrix(hstack([_zeros, -CWC_all]), number_type='float')
        F.rep_type = cdd.RepType.INEQUALITY
        # C * w_all + d == 0
        _d = d.reshape((C.shape[0], 1))
        F.extend(hstack([_d, C]), linear=True)
        P = cdd.Polyhedron(F)
        V = array(P.get_generators())
        poly = []
        for i in xrange(V.shape[0]):
            if V[i, 0] != 1:  # 1 = vertex, 0 = ray
                raise Exception("Not a polygon, V =\n%s" % repr(V))
            pG = dot(D, V[i, 1:])
            poly.append(pG)
        if all_plots:  # Check 1: plot COM trajectory and polygons
            plot_polygon(poly)
        if True:  # Check 2: using full H-representation
            # (autonomous but time consuming when designing the motion)
            self.check_all_inequalities(com_positions, poly)

    def check_all_inequalities(self, com_positions, poly):
        V = hstack([ones((len(poly), 1)), array(poly)])
        M = cdd.Matrix(V, number_type='float')
        M.rep_type = cdd.RepType.GENERATOR
        P = cdd.Polyhedron(M)
        H = array(P.get_inequalities())
        b, A = H[:, 0], H[:, 1:]
        for com in com_positions:
            if not all(dot(A, com[:2]) + b >= 0):
                raise Exception("Unstable CoM")

    def add_linear_com_objective(self, tracker, start_com, target_com, gain):
        self.check_com_positions([start_com, target_com])

        com_traj = LinearChunk.interpolate(
            start_com, target_com, T=tracker.duration)

        if all_plots:
            pylab.plot(
                [start_com[0], target_com[0]], [start_com[1], target_com[1]],
                'r--', lw=5)

        def vel_fun(t):
            # display_box(self.robot.env, target_com, color='g')
            return com_traj.qd(t)

        def jacobian(q):
            return self.robot.compute_com_jacobian(q)

        tracker.add_objective(VelocityConstraint(vel_fun, jacobian, gain),
                              weight=w_com)

    def add_linear_link_objective(self, tracker, link, start_pose, target_pose,
                                  gain):
        link_pos_traj = LinearChunk.interpolate(
            start_pose[4:], target_pose[4:], T=tracker.duration)

        def vel_fun(t):
            steering_pose = target_pose - link.GetTransformPose()
            steering_pose[4:] = link_pos_traj.qd(t)
            display_box(self.robot.env, target_pose[4:], color='b')
            return steering_pose

        def jacobian(q):
            return self.robot.compute_link_pose_jacobian(link, q)

        tracker.add_objective(VelocityConstraint(vel_fun, jacobian, gain))

    def move_com(self, via_com, gain, duration):
        chunk_duration = duration / len(via_com)
        for target_com in via_com:
            tracker = self.init_tracker(chunk_duration)
            com0 = self.cur_com
            com1 = target_com
            self.add_linear_com_objective(tracker, com0, com1, gain)
            new_chunk = tracker.track()
            self.chunks.append(new_chunk)
            self.cur_q = new_chunk.q(new_chunk.duration)

    def move_link_com(self, link, via_points, gain, duration):
        chunk_duration = duration / len(via_points)
        for i, (target_pose, target_com) in enumerate(via_points):
            print "Via point %d" % i
            tracker = self.init_tracker(chunk_duration)
            start_com = self.cur_com
            start_pose = self.robot.compute_link_pose(link, self.cur_q)
            self.add_linear_com_objective(tracker, start_com, target_com, gain)
            self.add_linear_link_objective(tracker, link, start_pose,
                                           target_pose, gain)
            new_chunk = tracker.track()
            self.chunks.append(new_chunk)
            self.cur_q = new_chunk.q(new_chunk.duration)
