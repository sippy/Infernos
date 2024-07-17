#!/bin/sh

set -e
set -x

. /opt/conda/etc/profile.d/conda.sh
conda create -y --name "${CONDA_MAINENV}" python=${PYTHON_VER}
conda activate "${CONDA_MAINENV}"
conda install -y pip
${PYTHON_CMD} -m pip install -U pip
echo "/opt/conda/envs/${CONDA_MAINENV}/lib" > "/etc/ld.so.conf.d/zzz-conda-${CONDA_MAINENV}.conf"
ldconfig
rm -r /opt/conda/pkgs
rm -r ~/.cache
mkdir ~/.cache
