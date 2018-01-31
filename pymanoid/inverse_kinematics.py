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


import cvxopt
import cvxopt.solvers
import os
import time

from pymanoid import interpolate
from pymanoid.rave import display_box
from pymanoid.trajectory import Trajectory, PolynomialChunk
from numpy import arange, minimum, maximum, zeros
from numpy import dot, vstack, hstack, array

from numpy import eye, ones
import cvxopt
import cvxopt.solvers

cvxopt.solvers.options['show_progress'] = False  # disable cvxopt output


def take_screenshot(robot):
    fname = './recording/%05d.png' % robot.frame_index
    os.system('import -window %s %s' % (robot.window_id, fname))
    robot.frame_index += 1
    time.sleep(1. / 30)


class Tracker(object):

    def __init__(self, robot, start_q, duration, dt, K_doflim, w_reg,
                 dof_lim_scale, vel_lim_scale):
        self.robot = robot
        self.start_q = start_q
        self.duration = duration
        self.dt = dt
        self.K_doflim = K_doflim
        self.w_reg = w_reg

        self.objectives = []
        self.constraints = []
        self.I = eye(self.robot.nb_dof)

        q_max = self.robot.q_max.copy()
        q_min = self.robot.q_min.copy()
        q_avg = .5 * (q_max + q_min)
        q_dev = .5 * (q_max - q_min)

        self.q_max = q_avg + dof_lim_scale * q_dev
        self.q_min = q_avg - dof_lim_scale * q_dev
        self.qd_max = +vel_lim_scale * ones(self.robot.nb_dof)
        self.qd_min = -vel_lim_scale * ones(self.robot.nb_dof)

    def add_objective(self, pos_constraint, weight=1.):
        self.objectives.append((weight, pos_constraint))

    def add_constraint(self, pos_constraint):
        self.constraints.append(pos_constraint)


class VelocityConstraint(object):

    def __init__(self, vel_fun, jacobian_fun, gain=None):
        self.vel = vel_fun
        self.gain = gain
        self.jacobian = jacobian_fun


class VelocityTracker(Tracker):

    def __init__(self, robot, start_q, duration, dt, K_doflim=10., w_reg=1e-3,
                 dof_lim_scale=0.95, vel_lim_scale=1000.):
        super(VelocityTracker, self).__init__(
            robot, start_q, duration, dt, K_doflim, w_reg, dof_lim_scale,
            vel_lim_scale)

    def fix_link(self, link, ref_q):
        self.add_constraint(VelocityConstraint(
            vel_fun=lambda t: zeros(6),
            jacobian_fun=lambda q: self.robot.compute_link_jacobian(link, q)))

    def attract_dof(self, dof, ref_value, gain=1., weight=0.005):
        J = zeros(len(self.start_q))
        J[dof.index] = 1.

        def vel_fun(t):
            q = self.robot.get_dof_values()
            return ref_value - q[dof.index]

        self.add_objective(VelocityConstraint(
            vel_fun=vel_fun,
            jacobian_fun=lambda q: J,
            gain=gain),
            weight=weight)

    def interpolate_com_linear(self, start_com, target_com, gain):
        com_traj = interpolate.linear(
            start_com, target_com, duration=self.duration)

        def vel_fun(t):
            display_box(self.robot.env, target_com, color='g')
            return com_traj.qd(t)

        return VelocityConstraint(
            vel_fun=vel_fun,
            jacobian_fun=lambda q: self.robot.compute_com_jacobian(q),
            gain=gain)

    def interpolate_link_linear(self, link, start_pose, target_pose, gain):
        link_pos_traj = interpolate.linear(
            start_pose[4:], target_pose[4:], duration=self.duration)

        def vel_fun(t):
            steering_pose = target_pose - link.GetTransformPose()
            steering_pose[4:] = link_pos_traj.qd(t)
            display_box(self.robot.env, target_pose[4:], color='b')
            return steering_pose

        def J(q):
            return self.robot.compute_link_pose_jacobian(link, q)

        return VelocityConstraint(
            vel_fun=vel_fun,
            jacobian_fun=J,
            gain=gain)

    def compute_instant_vel(self, t, q):
        qd_max = minimum(self.qd_max, self.K_doflim * (self.q_max - q))
        qd_min = maximum(self.qd_min, self.K_doflim * (self.q_min - q))
        J_list = [c.jacobian(q) for c in self.constraints]
        v_list = [c.vel(t) for c in self.constraints]
        P0 = self.w_reg * self.I
        q0 = zeros(len(q))

        for (w_obj, objective) in self.objectives:
            J = objective.jacobian(q)
            v = objective.vel(t)
            P0 += w_obj * dot(J.T, J)
            q0 += w_obj * dot(-v.T, J)

        qp_P = cvxopt.matrix(P0)
        qp_q = cvxopt.matrix(q0)
        qp_G = cvxopt.matrix(vstack([+self.I, -self.I]))
        qp_h = cvxopt.matrix(hstack([qd_max, -qd_min]))
        qp_A = cvxopt.matrix(vstack(J_list))
        qp_b = cvxopt.matrix(hstack(v_list))
        qp_x = cvxopt.solvers.qp(qp_P, qp_q, qp_G, qp_h, qp_A, qp_b)['x']
        qd = array(qp_x).reshape((self.robot.nb_dof,))
        return qd

    def track(self):
        record_video = hasattr(self.robot, 'window_id')
        q = self.start_q.copy()
        chunks = []

        for x in self.objectives:
            (w_obj, objective) = x if type(x) is tuple else (1., x)  # krooon
            if w_obj < self.w_reg:
                print "Warning: w_obj=%f < w_reg=%f" % (w_obj, self.w_reg)

        for t in arange(0, self.duration, self.dt):
            qd = self.compute_instant_vel(t, q)
            q = q + qd * self.dt
            chunk_poly = PolynomialChunk.from_coeffs([qd, q], self.dt)
            chunks.append(chunk_poly)
            if record_video and int(t / self.dt) % 6 == 0:
                take_screenshot(self.robot)
            self.robot.set_dof_values(q)  # needed for *.vel(t)
            self.robot.display_com(q)
            self.robot.display_floor_com(q)

        return Trajectory(chunks)


class AccelerationConstraint(object):

    def __init__(self, vel_fun, jacobian_fun, hessian_fun, gain=None):
        self.vel = vel_fun
        self.gain = gain
        self.jacobian = jacobian_fun
        self.hessian_term = hessian_fun


class AccelerationTracker(Tracker):

    def __init__(self, robot, start_q, start_qd, duration, dt, K_doflim=10.,
                 w_reg=1e-3, dof_lim_scale=0.95, vel_lim_scale=1000.):
        super(AccelerationTracker, self).__init__(
            robot, start_q, duration, dt, K_doflim, w_reg, dof_lim_scale,
            vel_lim_scale)
        self.start_qd = start_qd

    def fix_link(self, link, ref_q):
        self.add_constraint(AccelerationConstraint(
            vel_fun=lambda t: zeros(6),
            jacobian_fun=lambda q: self.robot.compute_link_jacobian(link, q),
            hessian_fun=lambda q, qd: dot(
                qd, dot(self.robot.compute_link_hessian(link, q), qd))))

    def attract_dof(self, dof, ref_value, gain=1., weight=0.005):
        zero_vel = zeros(1)
        J = zeros((1, len(self.start_q)))
        J[0][dof.index] = 1.

        def vel_fun(t):
            q = self.robot.get_dof_values()
            return ref_value - q[dof.index]

        self.add_objective(AccelerationConstraint(
            vel_fun=vel_fun,
            jacobian_fun=lambda q: J,
            hessian_fun=lambda q, qd: zero_vel,
            gain=gain),
            weight=weight)

    def interpolate_com_linear(self, start_com, target_com, gain):
        com_traj = interpolate.linear(
            start_com, target_com, duration=self.duration)

        def vel_fun(t):
            display_box(self.robot.env, target_com, color='g')
            return com_traj.qd(t)

        return AccelerationConstraint(
            vel_fun=vel_fun,
            gain=gain,
            jacobian_fun=lambda q: self.robot.compute_com_jacobian(q),
            hessian_fun=lambda q, qd: dot(
                qd, dot(self.robot.compute_com_hessian(q), qd)))

    def interpolate_link_linear(self, link, start_pose, target_pose, gain):
        link_pos_traj = interpolate.linear(
            start_pose[4:], target_pose[4:], duration=self.duration)

        def vel_fun(t):
            steering_pose = target_pose - link.GetTransformPose()
            steering_pose[4:] = link_pos_traj.qd(t)
            display_box(self.robot.env, target_pose[4:], color='b')
            return steering_pose

        def J(q):
            return self.robot.compute_link_pose_jacobian(link, q)

        def H_disc(q, qd):
            return (dot(J(q + qd * 1e-5) - J(q), qd)) / 1e-5

        return AccelerationConstraint(
            vel_fun=vel_fun,
            jacobian_fun=J,
            hessian_fun=H_disc,
            gain=gain)

    def compute_instant_acc(self, t, q, qd):
        qd_max = 10. * (self.q_max - q)
        qd_min = 10. * (self.q_min - q)
        qdd_max = 10. * (qd_max - qd)
        qdd_min = 10. * (qd_min - qd)

        A_list = []
        b_list = []

        qp_P = self.w_reg * self.I
        qp_q = zeros(len(q))

        for w_obj, obj in self.objectives:
            J = obj.jacobian(q)
            qd_H_qd = obj.hessian_term(q, qd)
            v = obj.vel(t)
            g = obj.gain
            qp_P += w_obj * dot(J.T, J)
            qp_q += w_obj * dot(-(g * (v - dot(J, qd)) - qd_H_qd).T, J)

        for cons in self.constraints:
            J = cons.jacobian(q)
            qd_H_qd = cons.hessian_term(q, qd)
            v = cons.vel(t)
            g = cons.gain
            A_list.append(J)
            b_list.append(g * (v - dot(J, qd)) + qd_H_qd)

        qp_P = cvxopt.matrix(qp_P)
        qp_q = cvxopt.matrix(qp_q)
        qp_G = cvxopt.matrix(vstack([+self.I, -self.I]))
        qp_h = cvxopt.matrix(hstack([qdd_max, -qdd_min]))
        qp_A = cvxopt.matrix(vstack(A_list))
        qp_b = cvxopt.matrix(hstack(b_list))
        qp_x = cvxopt.solvers.qp(qp_P, qp_q, qp_G, qp_h, qp_A, qp_b)['x']
        qdd = array(qp_x).reshape((self.robot.nb_dof,))
        return qdd

    def track(self):
        assert self.q_max is not None
        chunks = []
        q = self.start_q.copy()
        qd = self.start_qd.copy()
        dt = self.dt

        for w_obj, objective in self.objectives:
            if w_obj < self.w_reg:
                print "Warning: w_obj=%f < w_reg=%f" % (w_obj, self.w_reg)

        for t in arange(0, self.duration, dt):
            qdd = self.compute_instant_acc(t, q, qd)
            qd = qd + qdd * dt
            q = q + qd * dt + .5 * qdd * dt ** 2
            chunk_poly = PolynomialChunk.from_coeffs([.5 * qdd, qd, q], dt)
            chunks.append(chunk_poly)
            self.robot.set_dof_values(q)  # needed for *.vel(t)
            self.robot.display_com(q)
            self.robot.display_floor_com(q)

        return Trajectory(chunks)
