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
import pylab

from numpy import array, zeros, dot, hstack, vstack
from pymanoid.all_ik import VirtualTracker


cvxopt.solvers.options['show_progress'] = False


class ConvergenceFailed(Exception):

    pass


class IGConstraint(object):

    def __init__(self, diff_fun, jacobian_fun, gain):
        self.diff = diff_fun
        self.gain = gain
        self.jacobian = jacobian_fun


class IGTracker(VirtualTracker):

    def target_link_pos(self, link, target_pos, gain, link_coord=None):
        r = self.robot

        def f(q):
            p = r.compute_link_pos(link, q, link_coord)
            return p

        def J(q):
            return r.compute_link_translation_jacobian(link, q, link_coord)

        self.add_objective(IGConstraint(
            diff_fun=lambda q: target_pos - f(q),
            jacobian_fun=J,
            gain=gain))

    def target_link_pose(self, link, target_pose, gain):
        def f(q):
            return self.robot.compute_link_pose(link, q)

        def J(q):
            return self.robot.compute_link_pose_jacobian(link, q)

        self.add_objective(IGConstraint(
            diff_fun=lambda q: target_pose - f(q),
            jacobian_fun=J,
            gain=gain))

    def target_com(self, target_com, gain):
        def f(q):
            return self.robot.compute_com(q)

        def J(q):
            return self.robot.compute_com_jacobian(q)

        self.add_objective(IGConstraint(
            diff_fun=lambda q: target_com - f(q),
            jacobian_fun=J,
            gain=gain))

    def fix_link(self, link):
        def J(q):
            return self.robot.compute_link_jacobian(link, q)

        zeros6 = zeros(6)
        self.add_constraint(IGConstraint(
            diff_fun=lambda q: zeros6,
            jacobian_fun=J,
            gain=1.))

    def compute_instant_dq(self, q):
        dq_max = self.q_max - q
        dq_min = self.q_min - q

        J_list = [c.jacobian(q) for c in self.constraints]
        df_list = [c.gain * c.diff(q) for c in self.constraints]
        P0 = self.w_reg * self.I
        q0 = zeros(len(q))

        for (w_obj, objective) in self.objectives:
            J = objective.jacobian(q)
            df = objective.gain * objective.diff(q)
            P0 += w_obj * dot(J.T, J)
            q0 += w_obj * dot(-df.T, J)

        qp_P = cvxopt.matrix(P0)
        qp_q = cvxopt.matrix(q0)
        qp_G = cvxopt.matrix(vstack([+self.I, -self.I]))
        qp_h = cvxopt.matrix(hstack([+dq_max, -dq_min]))
        if J_list:
            qp_A = cvxopt.matrix(vstack(J_list))
            qp_b = cvxopt.matrix(hstack(df_list))
            qp_x = cvxopt.solvers.qp(qp_P, qp_q, qp_G, qp_h, qp_A, qp_b)['x']
        else:
            qp_x = cvxopt.solvers.qp(qp_P, qp_q, qp_G, qp_h)['x']
        dq = array(qp_x).reshape((self.robot.nb_dof,))
        return dq

    def compute_q(self, max_iter=100):
        def converged(q):
            n = array([pylab.norm(obj.diff(q)) for _, obj in self.objectives])
            return pylab.all(n < 1e-3)

        i, q = 0, self.start_q
        while not converged(q) and i < max_iter:
            i, q = i + 1, q + self.compute_instant_dq(q)

        if i >= max_iter:
            raise ConvergenceFailed()
        return q
