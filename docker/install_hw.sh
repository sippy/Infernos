#!/bin/sh

set -e
set -x

PIP_INSTALL="${PYTHON_CMD} -m pip install"

${CONDA_ACTIVATE}

case "${INFER_HW}" in
nvidia)
  ;;
intel)
  curl https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
   gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
   tee /etc/apt/sources.list.d/oneAPI.list
  ${APT_UPDATE}
  ${APT_INSTALL} libze1 ocl-icd-libopencl1
  ${APT_INSTALL} intel-oneapi-dpcpp-cpp-2024.1=2024.1.0-963 intel-oneapi-mkl-devel=2024.1.0-691
  apt-mark hold intel-oneapi-dpcpp-cpp-2024.1 intel-oneapi-mkl-devel
  ${PIP_INSTALL} torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 \
   intel-extension-for-pytorch==2.1.30.post0 oneccl_bind_pt==2.1.300+xpu \
   --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
  printf "/opt/intel/oneapi/mkl/2024.1/lib\n/opt/intel/oneapi/compiler/2024.1/lib\n" > \
   /etc/ld.so.conf.d/zzz-intel-oneapi.conf
  ldconfig
  ;;
*)
  echo "Unknown INFER_HW: '${INFER_HW}'" >&2
  false
  ;;
esac

rm -r ~/.cache
mkdir ~/.cache
