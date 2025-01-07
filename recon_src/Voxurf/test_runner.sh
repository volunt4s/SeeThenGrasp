#!/usr/bin/env bash

CONFIG=$1
WORKDIR=$2
SCENE=$3

echo python run.py --config ${CONFIG}/coarse.py -p ${WORKDIR} --no_reload --run_dvgo_init --sdf_mode voxurf_coarse --scene ${SCENE}
python3 run.py --config ${CONFIG}/coarse.py -p ${WORKDIR} --no_reload --run_dvgo_init --sdf_mode voxurf_coarse --scene ${SCENE}
