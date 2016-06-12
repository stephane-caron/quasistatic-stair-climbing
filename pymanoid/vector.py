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


from numpy import dot, sqrt


def norm(vector):
    """Two times faster than pylab.norm:

        In [1]: %timeit norm(random.random(42))
        100000 loops, best of 3: 6.77 us per loop

        In [2]: %timeit pylab.norm(random.random(42))
        100000 loops, best of 3: 14.1 us per loop

    """
    return sqrt(dot(vector, vector))
