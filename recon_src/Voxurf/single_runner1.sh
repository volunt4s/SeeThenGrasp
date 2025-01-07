#!/usr/bin/env bash

CONFIG=$1
WORKDIR=$2
SCENE=$3

echo python run.py --config ${CONFIG}/coarse.py -p ${WORKDIR} --no_reload --run_dvgo_init --sdf_mode voxurf_coarse --scene ${SCENE}
python3 run_1.py --config ${CONFIG}/coarse.py -p ${WORKDIR} --no_reload --run_dvgo_init --sdf_mode voxurf_coarse --scene ${SCENE}

echo python run.py --config ${CONFIG}/fine.py --render_train -p ${WORKDIR} --no_reload --sdf_mode voxurf_fine --scene ${SCENE}
python3 run_1.py --config ${CONFIG}/fine.py --render_train -p ${WORKDIR} --no_reload --sdf_mode voxurf_fine --scene ${SCENE}

#python3 run.py --config ${CONFIG}/fine.py -p ${WORKDIR} --sdf_mode voxurf_fine --scene ${SCENE} --render_only --render_poses

python3 run_1.py --config ${CONFIG}/fine.py -p ${WORKDIR} --sdf_mode voxurf_fine --scene ${SCENE} --render_only --mesh_from_sdf
