#!/bin/sh
source /usr/share/Modules/init/bash
unset MODULEPATH
module use /lustre/usr/modulefiles/pi
module purge
module load gcc/4.8
#source ~/users/htl11/bashrc_htl11

#bash $1
python btm.py
