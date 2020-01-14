#!/bin/sh

gnome-terminal -- bash -c "/home/mikami-lab/anaconda3/bin/python train.py --env-name DobotPush-v1 --learning-from $1 --n-skip-frames 10 --cuda; bash"

