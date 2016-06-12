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


import pymanoid
import openravepy

from pymanoid.bodies import PseudoFoot
from pymanoid.pointsets import Rectangles
from pymanoid.rotation import quat_from_rpy
from pymanoid.trajectory import Chunk
from pymanoid.trajectory import Trajectory
from pymanoid.sketch import TrajectorySketch
from numpy import array, dot, hstack, pi


rect_file = 'data/exp1/rectangles.txt'  # path to geometric data file
post_contact_dz = 0.0
export_slowdown = 2.


def export_sketch(sketch, segment_id):
    def fix_wrists(q):
        q[45:48] = 0.
        q[36:39] = 0.
        return q

    fname = 'motion-segment%d' % segment_id
    traj_sketch = sketch.get_trajectory()
    traj_fixed = Trajectory([Chunk(
        T=traj_sketch.duration,
        q=lambda t: fix_wrists(traj_sketch.q(t)),
        qd=lambda t: fix_wrists(traj_sketch.qd(t)),
        qdd=lambda t: fix_wrists(traj_sketch.qdd(t)))])
    traj = traj_fixed.timescale(export_slowdown)
    hrp.write_pos_file(traj, fname)
    return traj


def segment_1():
    print "Computing segment 1..."
    sketch = TrajectorySketch(hrp, hrp.get_dof_values())
    sketch.contact_link(hrp.left_foot)
    sketch.contact_link(hrp.right_foot)

    com_in_lfoot_frame = hrp.left_foot_center_from_origin \
        + array([0., -0.01, 0.75])
    left_com = dot(hrp.left_foot.GetTransform(),
                   hstack([com_in_lfoot_frame, [1]]))[:3]
    sketch.move_com([left_com], gain=10., duration=2.)

    traj = export_sketch(sketch, segment_id=1)
    return traj


def segment_2():
    print "Computing segment 2..."
    sketch = TrajectorySketch(hrp, hrp.get_dof_values())
    sketch.contact_link(hrp.left_foot)

    start_com = sketch.cur_com
    start_rfoot = hrp.right_foot.GetTransformPose()[4:]
    end_rfoot = hrp.origin_for_target(hrp.right_foot, right_step)[4:]
    via_points = [
        # Via point 0
        (hstack([quat_from_rpy(0, 0, 0),
                 start_rfoot + [0.0, 0., 0.1]]),
            start_com + array([0.0, 0.0, 0.02])),
        # Via point 1
        (hstack([quat_from_rpy(0, -pi / 4, 0),
                 start_rfoot + [0.0, -0.02, 0.25]]),
            start_com + array([0.0, 0.0, 0.04])),
        # Via point 2
        (hstack([[1., 0., 0., 0.],
                 end_rfoot + array([0., 0., +0.03])]),
            start_com + array([0.01, 0.0, 0.05])),
        # Via point 3
        (hstack([[1., 0., 0., 0.],
                 end_rfoot + [0., 0., -0.01]]),
            start_com + array([0.02, 0.0, 0.035]))]
    sketch.move_link_com(hrp.right_foot, via_points, gain=15., duration=9.)

    traj = export_sketch(sketch, segment_id=2)
    return traj


def segment_3():
    print "Computing segment 3..."
    q_upper_ref = hrp.q_halfsit.copy()
    q_upper_ref[hrp.R_SHOULDER_P.index] = -1
    q_upper_ref[hrp.L_SHOULDER_P.index] = -1
    q_upper_ref[hrp.R_ELBOW_P.index] = -0.9
    q_upper_ref[hrp.L_ELBOW_P.index] = -1.2

    sketch = TrajectorySketch(hrp, hrp.get_dof_values(),
                              q_upper_ref=q_upper_ref)
    sketch.contact_link(hrp.left_foot)
    sketch.contact_link(hrp.right_foot)

    start_com = sketch.cur_com
    via_com = [
        start_com + array([0., 0., 0.]),  # for q_upper_ref
        start_com + array([0.1, -0.10, 0.05]),
        start_com + array([0.5, -0.18, 0.05])]
    sketch.move_com(via_com, gain=7., duration=10.)

    traj = export_sketch(sketch, segment_id=3)
    return traj


def segment_4():
    print "Computing segment 4..."
    q_upper_ref = hrp.q_halfsit.copy()
    q_upper_ref[hrp.R_SHOULDER_P.index] = -1
    q_upper_ref[hrp.L_SHOULDER_P.index] = -1
    q_upper_ref[hrp.R_ELBOW_P.index] = -0.9
    q_upper_ref[hrp.L_ELBOW_P.index] = -1.2
    sketch = TrajectorySketch(hrp, hrp.get_dof_values(),
                              q_upper_ref=q_upper_ref)
    sketch.contact_link(hrp.right_foot)

    start_com = sketch.cur_com
    start_lfoot = hrp.left_foot.GetTransformPose()[4:]
    end_lfoot = hrp.origin_for_target(hrp.left_foot, left_step)[4:]
    via_points = [
        # Via point 0
        (hstack([quat_from_rpy(0, pi / 4, 0),
                 start_lfoot + [0.0, 0., 0.1]]),
         start_com + array([0.05, -0.02, 0.15])),
        # Via point 1
        (hstack([quat_from_rpy(0, -pi / 4, 0),
                 start_lfoot + [-0.05, 0., 0.3]]),
         start_com + array([0.05, -0.01, 0.2])),
        # Via point 2
        (hstack([[1., 0., 0., 0.],
                 end_lfoot + [0., 0., +0.06]]),
         start_com + array([0.05, 0.0, 0.18])),
        # Via point 3
        (hstack([[1., 0., 0., 0.],
                 end_lfoot + [0., 0., 0.0]]),
         start_com + array([0.05, 0.0, 0.16]))]
    sketch.move_link_com(hrp.left_foot, via_points, gain=4., duration=8.)

    traj = export_sketch(sketch, segment_id=4)
    return traj


def segment_5():
    print "Computing segment 5..."
    sketch = TrajectorySketch(hrp, hrp.get_dof_values())
    sketch.contact_link(hrp.right_foot)
    sketch.contact_link(hrp.left_foot)

    sketch.move_com([sketch.cur_com + array([0., +0.05, 0.])],
                    gain=10., duration=2.)

    traj = export_sketch(sketch, segment_id=5)
    return traj


def step(rectangle):
    global left_pose, right_pose
    global left_step, right_step
    chunks = []
    left_pose[4] = rectangle.x
    left_pose[6] = rectangle.z + 0.025
    left_step.set_transform_pose(left_pose)
    right_pose[4] = rectangle.x
    right_pose[6] = rectangle.z + 0.025
    right_step.set_transform_pose(right_pose)
    chunks.append(segment_1())
    chunks.append(segment_2())
    chunks.append(segment_3())
    chunks.append(segment_4())
    chunks.append(segment_5())
    return Trajectory(chunks)


if __name__ == "__main__":
    segments = []

    env = openravepy.Environment()
    env.Load('env.xml')

    hrp = pymanoid.hrp4.HRP4(env)
    hrp.q_max[hrp.R_SHOULDER_R.index] = -0.1
    hrp.q_min[hrp.L_SHOULDER_R.index] = +0.1
    hrp.q_max[hrp.R_KNEE_P.index] = 80. * pi / 180.
    hrp.set_transparency(0)

    env.SetViewer('qtcoin')
    viewer = env.GetViewer()
    cam_trans = array([
        [0.2936988, -0.51198474, 0.80722527, -1.43207502],
        [-0.95587863, -0.16268234, 0.24460276, -0.6278013],
        [0.00608842, -0.84344892, -0.53717488, 1.86361814],
        [0., 0., 0., 1.]])
    cam_trans = array([
        [0.95410322, -0.045319, 0.2960291, -0.17090011],
        [-0.29879035, -0.21099493, 0.9307016, -2.66766453],
        [0.02028217, -0.97643603, -0.21485182, 1.20659459],
        [0., 0., 0., 1.]])
    viewer.SetBkgndColor([0.9, 0.9, 0.9])
    viewer.SetBkgndColor([0.5, 0.6, 0.9])
    viewer.SetCamera(cam_trans)

    rectangles = Rectangles(env, rect_file)
    q_start = hrp.q_halfsit.copy()
    q_start[hrp.TRANS_X.index] -= 0.15
    hrp.set_dof_values(q_start)
    left_pose = hrp.left_foot.GetTransformPose()
    left_pose[6] -= 0.01 + 0.06
    right_pose = hrp.right_foot.GetTransformPose()
    right_pose[5] = left_pose[5] - 0.18  # fixed inter-foot distance
    right_pose[6] = left_pose[6]

    show_footsteps = []
    tmp_left_pose = left_pose.copy()
    tmp_right_pose = right_pose.copy()
    for i, rectangle in enumerate(rectangles.rectangles):
        tmp_left_pose[4] = rectangle.x
        tmp_left_pose[6] = rectangle.z + 0.025
        tmp_right_pose[4] = rectangle.x
        tmp_right_pose[6] = rectangle.z + 0.025
        show_footsteps.append(PseudoFoot(
            env, 'LeftStep%d' % i, pose=tmp_left_pose, color='g',
            transparency=0.2))
        show_footsteps.append(PseudoFoot(
            env, 'RightStep%d' % i, pose=tmp_right_pose, color='g',
            transparency=0.2))

    left_step = PseudoFoot(env, 'LeftStep', pose=left_pose)
    right_step = PseudoFoot(env, 'RightStep', pose=right_pose, color='r')
    hrp.set_dof_values(q_start)

    chunks = [step(rect) for rect in rectangles.rectangles]
    traj = Trajectory(chunks)

    import IPython
    IPython.embed()
