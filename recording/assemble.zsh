#!/bin/zsh

FRAMERATE=33  # assumes dt=3e-2 [s]
EXT=mp4

avconv -r ${FRAMERATE} -qscale 1 -i ./%05d.png ./video.${EXT}
