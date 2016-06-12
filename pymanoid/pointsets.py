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


import itertools

from pymanoid.bodies import Box
from numpy import array, cos, cross, dot, int64, sin, vstack
from pylab import norm
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull


def draw_polyhedron(env, points, color=None, plot_type=6, precomp_hull=None,
                    linewidth=1., pointsize=0.02):
    """
    Draw a polyhedron defined as the convex hull of a set of points.

    env -- openravepy environment
    points -- list of 3D points
    color -- RGBA vector
    plot_type -- bitmask with 1 for vertices, 2 for edges and 4 for surface
    precomp_hull -- used in the 2D case where the hull has zero volume
    linewidth -- openravepy format
    pointsize -- openravepy format

    """
    is_2d = precomp_hull is not None
    hull = precomp_hull if precomp_hull is not None else ConvexHull(points)
    vertices = array([points[i] for i in hull.vertices])
    points = array(points)
    color = array(color if color is not None else (0., 0.5, 0., 1.))
    handles = []
    if plot_type & 2:  # include edges
        edge_color = color * 0.7
        edge_color[3] = 1.
        edges = vstack([[points[i], points[j]]
                        for s in hull.simplices
                        for (i, j) in itertools.combinations(s, 2)])
        edges = array(edges)
        handles.append(env.drawlinelist(edges, linewidth=linewidth,
                                        colors=edge_color))
    if plot_type & 4:  # include surface
        if is_2d:
            nv = len(vertices)
            indices = array([(0, i, i + 1) for i in xrange(nv - 1)], int64)
            handles.append(env.drawtrimesh(vertices, indices, colors=color))
        else:
            indices = array(hull.simplices, int64)
            handles.append(env.drawtrimesh(points, indices, colors=color))
    if plot_type & 1:  # vertices
        color[3] = 1.
        handles.append(env.plot3(vertices, pointsize=pointsize, drawstyle=1,
                                 colors=color))
    return handles


def draw_polygon(env, points, n=None, color=None, plot_type=3, linewidth=1.,
                 pointsize=0.02):
    """
    Draw a polygon defined as the convex hull of a set of points. The normal
    vector n of the plane containing the polygon must also be supplied.

    env -- openravepy environment
    points -- list of 3D points
    n -- plane normal vector
    color -- RGBA vector
    plot_type -- bitmask with 1 for edges, 2 for surfaces and 4 for summits
    linewidth -- openravepy format
    pointsize -- openravepy format

    """
    assert n is not None, "Please provide the plane normal as well"
    t1 = array([n[2] - n[1], n[0] - n[2], n[1] - n[0]], dtype=float)
    t1 /= norm(t1)
    t2 = cross(n, t1)
    points2d = [[dot(t1, x), dot(t2, x)] for x in points]
    hull = ConvexHull(points2d)
    return draw_polyhedron(env, points, color, plot_type, hull, linewidth,
                           pointsize)


class PointSet(object):

    params = {
        'exp1': {
            'dx': -0.7,
            'dy': 0.,
            'dz': 1.35,
            'pitch': -0.03
        },
        'exp2': {
            'dx': -0.7,
            'dy': 0.,
            'dz': 1.6,
            'pitch': -0.03
        }
    }

    def __init__(self, env, fname):
        print fname
        assert 'exp1' in fname or 'exp2' in fname
        p = 'exp1' if 'exp1' in fname else 'exp2'
        self.env = env
        self.dx = self.params[p]['dx']
        self.dy = self.params[p]['dy']
        self.dz = self.params[p]['dz']
        self.pitch = self.params[p]['pitch']
        self.points = []
        with open(fname, 'r') as f:
            for line in f:
                v = map(float, line.split(','))
                with self.env:
                    x = -cos(self.pitch) * v[1] + sin(self.pitch) * v[2]
                    y = v[0]
                    z = -sin(self.pitch) * v[1] - cos(self.pitch) * v[2]
                    x, y, z = y, -x, -z
                    x += self.dx
                    y += self.dy
                    z += self.dz
                    if x < -0.5 or abs(y) > 1.3:
                        continue
                    self.points.append([x, y, z])
        print "%d supervoxel centers" % len(self.points)


class SupervoxelTree(PointSet):

    def __init__(self, env, fname):
        super(SupervoxelTree, self).__init__(env, fname)
        dims = [0.05] * 3 if 'super' in fname else [0.02] * 3
        self.boxes = []
        for (x, y, z) in self.points:
            name = 'sv%d' % len(self.env.GetBodies())
            self.boxes.append(Box(self.env, name, dims=dims,
                                  pose=[1., 0., 0., 0., x, y, z], color='r'))
        self.tree = KDTree([box.p[:2] for box in self.boxes])

    def query(self, p, radius):
        indexes = self.tree.query_ball_point(p[0:2], radius)
        return [self.boxes[i] for i in indexes]


class Rectangles(PointSet):

    class Rectangle(object):

        def __init__(self, env, p0, p1, p2, p3):
            self.handle = draw_polygon(env, [p0, p1, p2, p3], [0, 0, 1],
                                       color=(0.6, 0.4, 0., 1.), plot_type=6,
                                       linewidth=5.)
            self.x = (p0[0] + p1[0] + p2[0] + p3[0]) / 4
            self.y = (p0[1] + p1[1] + p1[1] + p3[1]) / 4
            self.z = (p0[2] + p1[2] + p2[2] + p3[2]) / 4

    def __init__(self, env, fname):
        super(Rectangles, self).__init__(env, fname)
        assert len(self.points) % 4 == 0
        rectangles = []
        for i in xrange(len(self.points) / 4):
            p0, p1 = self.points[4 * i + 0], self.points[4 * i + 1]
            p2, p3 = self.points[4 * i + 2], self.points[4 * i + 3]
            rectangles.append(self.Rectangle(env, p0, p1, p2, p3))
        self.rectangles = sorted(rectangles, key=lambda r: r.z)
