# Supervoxel Plane Segmentation and Multi-Contact Motion Generation for Humanoid Stair Climbing

Source code for the motion-planning part of [this paper](https://scaron.info/research/ijhr-2016.html).

## Abstract

Stair climbing is still a challenging task for humanoid robots, especially in
unknown environments. In this paper, we address this problem from perception to
execution. Our first contribution is a real-time plane segment estimation
method using unorganized lidar data *without* prior models of the staircase. We
then integrate this solution with humanoid motion planning. Our second
contribution is a stair-climbing motion generator where estimated plane
segments are used to compute footholds and stability polygons. We evaluate our
method on various staircases. We also demonstrate the feasibility of the
generated trajectories in a real-life experiment with the humanoid robot HRP-4. 

<img src="https://scaron.info/images/ijhr-2016.png" width="500" align="center" />

Authors:
[Zhang Tianwei](http://zhangtianwei.info/),
[Stéphane Caron](https://scaron.info) and
[Yoshihiko Nakamura](http://www.ynl.t.u-tokyo.ac.jp/).

## Requirements

- [CVXOPT](http://cvxopt.org/) for quadratic programming
- [OpenRAVE](https://github.com/rdiankov/openrave) for forward kinematics and
  visualization ([installation instructions](https://scaron.info/teaching/installing-openrave-on-ubuntu-14.04.html))
- ``HRP4R.dae`` COLLADA model for HRP4 (md5sum: ``bb009f37a1783e3b029a77acb6b92a28``)
- ``pymanoid/hrp4.py`` (md5sum: ``82298e46b4d30106aff942dc6e867dfc``)

Unfortunately it is unclear whether we can release the last two files due to
copyright.

## Usage

Run ``python plan_motion.py`` from the top-level directory.

## Later work

See also [this work
(2018)](https://scaron.info/publications/stair-climbing.html) for some dynamic
stair climbing.
