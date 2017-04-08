#!/bin/sh
K=50
F="stop"
job_name="btm-k$K-f$F"
script_name="run.k$K.f$F.sh"
echo "sbatch --job-name=$job_name -n 16 -p cpu --output=../log/$job_name.log ./.run.sh $script_name"
sbatch --job-name=$job_name -n 16 -p cpu --output=~/users/htl11/topic-model/btm/log/$job_name.log ./.run.sh $script_name
