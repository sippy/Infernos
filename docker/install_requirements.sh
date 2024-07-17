#!/bin/sh

set -e
set -x

DEV_PKGS="cmake pkg-config make git patch"
PIP_INSTALL="${PYTHON_CMD} -m pip install"

if [ "${INFER_HW}" != "intel" ]
then
  DEV_PKGS="${DEV_PKGS} gcc g++ libc6-dev"
fi

${APT_INSTALL} ${DEV_PKGS}
${CONDA_ACTIVATE}

${PIP_INSTALL} -r requirements.txt

if [ "${INFER_HW}" = "intel" ]
then
  patch -d "/opt/conda/envs/${CONDA_MAINENV}/lib/python${PYTHON_VER}/site-packages" \
   -p2 -s < intel-ray.diff
  find "/opt/conda" -name "libstdc++.so.6*" -delete
fi

apt-get remove -y ${DEV_PKGS}
apt-get autoremove -y
rm -r ~/.cache
mkdir ~/.cache
