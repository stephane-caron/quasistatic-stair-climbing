# Quasi-static stair climbing

Source code for the motion-planning part of the paper [Supervoxel Plane Segmentation and Multi-Contact Motion Generation for Humanoid Stair Climbing](https://scaron.info/research/ijhr-2016.html).

<img src="https://scaron.info/images/ijhr-2016.png" width="500" align="center" />

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
