#!/usr/bin/env bash
CONFIG=$1
WORKDIR=$2
SCENE=$3

python run.py --config ${CONFIG}/fine.py -p ${WORKDIR} --sdf_mode voxurf_fine --scene ${SCENE} --render_only --mesh_from_sdf --extract_color
