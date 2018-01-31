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

import bisect
import numpy
import pymanoid

from numpy import poly1d, polyder, zeros


class NoTrajectoryFound(Exception):

    pass


class TrajectoryError(Exception):

    def __init__(self, msg, traj=None, t=None):
        self.msg = msg
        self.traj = traj
        self.t = None

    def __str__(self):
        if self.t is not None:
            return self.msg + " at time t=%f" % self.t
        return self.msg


class VirtualTrajectory(object):

    def __init__(self):
        self.duration = None
        self.q = None
        self.qd = None
        self.qdd = None

    @property
    def last_q(self):
        return self.q(self.duration)

    @property
    def last_qd(self):
        return self.qd(self.duration)

    def plot_q(self):
        import pylab
        trange = pylab.linspace(0., self.duration, 100)
        pylab.plot(trange, [self.q(t) for t in trange])

    def plot_qd(self):
        import pylab
        trange = pylab.linspace(0., self.duration, 100)
        pylab.plot(trange, [self.qd(t) for t in trange])

    def split(self, tlist):
        raise NotImplementedError()

    def timescale(self, scaling):
        raise NotImplementedError()


class Chunk(object):

    def __init__(self, T, q, qd, qdd=None):
        """Constructor.

        T -- duration
        q -- position function t -> q(t)
        qd -- velocity function t -> qd(t)
        qdd -- (optional) acceleration function t -> qdd(t)

        """
        self.T = T
        self.duration = T
        self.q = q
        self.qd = qd
        self.qdd = qdd

    @property
    def q_beg(self):
        return self.q(0)

    @property
    def qd_beg(self):
        return self.qd(0)

    @property
    def qdd_beg(self):
        return self.qdd(0)

    @property
    def q_end(self):
        return self.q(self.T)

    @property
    def qd_end(self):
        return self.qd(self.T)

    @property
    def qdd_end(self):
        return self.qdd(self.T)

    def retime(self, T2, s, sd, sdd):
        def q2(t):
            return self.q(s(t))

        def qd2(t):
            return sd(t) * self.qd(s(t))

        if not self.qdd:
            return Chunk(T2, q2, qd2)

        def qdd2(t):
            return sdd(t) * self.qd(s(t)) + sd(t) ** 2 * self.qdd(s(t))

        return Chunk(T2, q2, qd2, qdd2)

    def timescale(self, scaling):
        T2 = scaling * self.T
        s = poly1d([1. / scaling, 0])
        sd = s.deriv(1)
        sdd = s.deriv(2)
        return self.retime(T2, s, sd, sdd)


class LinearChunk(Chunk):

    def __init__(self, T, c1, c0):
        """Linear chunk: q(t) = c1 t + c0."""
        zz = zeros(len(c0))
        self.T = T
        self.c1 = c1
        self.c0 = c0
        self.q = lambda t: c1 * t + c0
        self.qd = lambda t: c1
        self.qdd = lambda t: zz

    @staticmethod
    def interpolate(q0, q1, T=None):
        traj = LinearChunk(1., q1 - q0, q0)
        if T is not None:
            return traj.timescale(T)
        return traj

    def timescale(self, scaling):
        T2 = scaling * self.T
        d1 = self.c1 / scaling
        d0 = self.c0
        return LinearChunk(T2, d1, d0)


class QuadraticChunk(Chunk):

    def __init__(self, T, c2, c1, c0):
        """Second-order polynomial chunk: q(t) = a t^2 + b t + c."""
        self.T = T
        self.c2 = c2
        self.c1 = c1
        self.c0 = c0
        self.q = lambda t: c2 * t ** 2 + c1 * t + c0
        self.qd = lambda t: 2 * c2 * t + c1
        self.qdd = lambda t: 2 * c2

    @staticmethod
    def interpolate(q0, q1, qd0=None, qd1=None, T=None):
        assert (qd0 is not None) != (qd1 is not None)
        if qd0 is not None:
            c2 = q1 - q0 - qd0
            c1 = qd0
            c0 = q0
        elif qd1 is not None:
            Delta_q = q1 - q0
            c2 = qd1 - Delta_q
            c1 = -qd1 + 2 * Delta_q
            c0 = q0
        traj = QuadraticChunk(1., c2, c1, c0)
        if T is not None:
            return traj.timescale(T)
        return traj

    def timescale(self, scaling):
        T2 = scaling * self.T
        d2 = self.c2 / scaling ** 2
        d1 = self.c1 / scaling
        d0 = self.c0
        return QuadraticChunk(T2, d2, d1, d0)


class CubicChunk(Chunk):

    def __init__(self, T, c3, c2, c1, c0):
        """Third-order polynomial chunk: q(t) = c3 t^3 + c2 t^2 + c1 t + c0."""
        self.T = T
        self.c3 = c3
        self.c2 = c2
        self.c1 = c1
        self.c0 = c0
        self.q = lambda t: c3 * t ** 3 + c2 * t ** 2 + c1 * t + c0
        self.qd = lambda t: 3 * c3 * t ** 2 + 2 * c2 * t + c1
        self.qdd = lambda t: 6 * c3 * t + 2 * c2

    @staticmethod
    def interpolate(q0, qd0, q1, qd1, T=None):
        """Bezier interpolation between (q0, qd0) and (q1, qd1)."""
        v1 = q0 + qd0 / 3.
        v2 = q1 - qd1 / 3.
        c3 = q1 - q0 + 3 * (v1 - v2)
        c2 = 3 * (v2 - 2 * v1 + q0)
        c1 = 3 * (v1 - q0)
        c0 = q0
        traj = CubicChunk(1., c3, c2, c1, c0)
        if T is not None:
            return traj.timescale(T)
        return traj

    def timescale(self, scaling):
        T2 = scaling * self.T
        d3 = self.c3 / scaling ** 3
        d2 = self.c2 / scaling ** 2
        d1 = self.c1 / scaling
        d0 = self.c0
        return CubicChunk(T2, d3, d2, d1, d0)


class ChunkPrev(VirtualTrajectory):

    def __init__(self, duration, q_fun, qd_fun, qdd_fun=None):
        self.duration = duration
        self.q = q_fun
        self.qd = qd_fun
        self.qdd = qdd_fun

    def timescale(self, scaling):
        s_poly = poly1d([1. / scaling, 0])
        new_duration = scaling * self.duration
        assert abs(s_poly(0.)) < 1e-10
        assert abs(s_poly(new_duration) - self.duration) < 1e-10
        s, sd, sdd = s_poly, s_poly.deriv(1), s_poly.deriv(2)

        def q(t):
            return self.q(s(t))

        def qd(t):
            return sd(t) * self.qd(s(t))

        def qdd(t):
            return sdd(t) * self.qd(s(t)) + sd(t) ** 2 * self.qdd(s(t))

        return Chunk(new_duration, q, qd, qdd_fun=qdd)


class PolynomialChunk(VirtualTrajectory):

    def __init__(self, duration, q_polynoms):
        qd_polynoms = [polyder(P) for P in q_polynoms]
        qdd_polynoms = [polyder(P) for P in qd_polynoms]
        self.q_polynoms = q_polynoms
        self.qd_polynoms = qd_polynoms
        self.qdd_polynoms = qdd_polynoms
        self.duration = duration
        self.q = lambda t: numpy.array([q(t) for q in q_polynoms])
        self.qd = lambda t: numpy.array([qd(t) for qd in qd_polynoms])
        self.qdd = lambda t: numpy.array([qdd(t) for qdd in qdd_polynoms])

    @staticmethod
    def from_coeffs(coeffs, duration):
        size = len(coeffs[0])
        indexes = xrange(size)
        assert all([len(v) == size for v in coeffs])
        q_polynoms = [poly1d([v[i] for v in coeffs]) for i in indexes]
        return PolynomialChunk(duration, q_polynoms)

    def split(self, tlist):
        out_chunks = []
        t0 = tlist[0]
        out_chunks.append(PolynomialChunk(t0, self.q_polynoms))
        tlist.append(self.duration)
        for t1 in tlist[1:]:
            duration2 = t1 - t0

            def shift(P):
                return pymanoid.poly1d.translate_zero(P, t0)

            q_polynoms2 = map(shift, self.q_polynoms)
            out_chunks.append(PolynomialChunk(duration2, q_polynoms2))
            t0 = t1
        return out_chunks

    def timescale(self, scaling):
        def timescale_poly(P):
            return P(poly1d([1. / scaling, 0]))

        return PolynomialChunk(
            self.duration * scaling,
            [timescale_poly(q) for q in self.q_polynoms])


class Trajectory(VirtualTrajectory):

    def __init__(self, chunks):
        dtns = [traj.duration for traj in chunks]
        self.chunks = chunks
        self.cum_durations = [sum(dtns[0:i]) for i in xrange(len(chunks) + 1)]
        self.duration = sum(dtns)
        self.nb_chunks = len(chunks)

    def chunk_at(self, t, return_chunk_index=False):
        i = bisect.bisect(self.cum_durations, t)
        assert i > 0, "The first cumulative time should be zero..."
        chunk_index = min(self.nb_chunks, i) - 1
        t_start = self.cum_durations[chunk_index]
        chunk = self.chunks[chunk_index]
        if return_chunk_index:
            return chunk, (t - t_start), chunk_index
        return chunk, (t - t_start)

    def q(self, t):
        chunk, t2 = self.chunk_at(t)
        return chunk.q(t2)

    def qd(self, t):
        chunk, t2 = self.chunk_at(t)
        return chunk.qd(t2)

    def qdd(self, t):
        chunk, t2 = self.chunk_at(t)
        return chunk.qdd(t2)

    def split3(self, t1, t2):
        """Split trajectory in three chunks.

        t1 -- first chunk time range is [0, t1]
        t2 -- second chunk time range is [t1, t2]

        """
        chunk1, tc1, i1 = self.chunk_at(t1, return_chunk_index=True)
        chunk2, tc2, i2 = self.chunk_at(t2, return_chunk_index=True)
        if i1 == i2:
            c1, c2, c4 = chunk1.split([tc1, tc2])
            c3 = []
        else:
            c1, c2 = chunk1.split([tc1])
            c3, c4 = chunk2.split([tc2])
        chunks_left = self.chunks[:i1] + [c1]
        chunks_mid = [c2] + self.chunks[i1:i2] + [c3]
        chunks_right = [c4] + self.chunks[i2:]
        traj_left = Trajectory(chunks_left)
        traj_mid = Trajectory(chunks_mid)
        traj_right = Trajectory(chunks_right)
        return traj_left, traj_mid, traj_right

    def timescale(self, scaling):
        f = Chunk(self.duration, self.q, self.qd, self.qdd)
        return Trajectory([f.timescale(scaling)])

    def deep_timescale(self, scaling):
        return Trajectory([chunk.timescale(scaling) for chunk in self.chunks])

    def retime(self, s, new_duration):
        assert abs(s(0)) < 1e-2
        print ""
        print "retiming ends at %f v. T=%f" % (s(new_duration), self.duration)
        print ""
        assert False, "mais cette fonction est fausse !"
        return Trajectory([Chunk(
            duration=new_duration,
            q_fun=lambda t: self.q(s(t)),
            qd_fun=lambda t: self.qd(s(t)),
            qdd_fun=lambda t: self.qdd(s(t)))])
