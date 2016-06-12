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


import vector

from trajectory import PolynomialChunk
from numpy import poly1d


def poly2(q0, q1, qd0=None, qd1=None, **kwargs):
    dofs = xrange(q0.shape[0])
    if qd0 is not None:
        C0, C1, C2 = q0, qd0, q1 - q0 - qd0
    elif qd1 is not None:
        Delta_q = vector.center_angle_vect(q1 - q0)
        C0, C1, C2 = q0, -qd1 + 2 * Delta_q, qd1 - Delta_q
    else:
        raise Exception("please provide either qd0 or qd1")
    q_polynoms = [poly1d([C2[i], C1[i], C0[i]]) for i in dofs]
    traj = PolynomialChunk(1., q_polynoms, **kwargs)
    return traj


def linear(q_init, q_dest, duration=None):
    n = len(q_init)
    C0 = q_init
    C1 = q_dest - q_init
    q_polynoms = [poly1d([C1[i], C0[i]]) for i in xrange(n)]
    traj = PolynomialChunk(1., q_polynoms)
    if duration is not None:
        return traj.timescale(duration)
    return traj


def bezier(q_init, qd_init, q_dest, qd_dest, duration=None):
    n = len(q_init)
    Delta_q = vector.center_angle_vect(q_dest - q_init)
    q0 = q_init
    q3 = q_init + Delta_q
    q1 = q0 + qd_init / 3.
    q2 = q3 - qd_dest / 3.
    C0 = q0
    C1 = 3 * (q1 - q0)
    C2 = 3 * (q2 - 2 * q1 + q0)
    C3 = -q0 + 3 * q1 - 3 * q2 + q3
    q_polynoms = [poly1d([C3[i], C2[i], C1[i], C0[i]]) for i in xrange(n)]
    traj = PolynomialChunk(1., q_polynoms)
    if duration is not None:
        return traj.timescale(duration)
    return traj
