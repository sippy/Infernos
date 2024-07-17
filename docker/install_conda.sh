#!/bin/sh

set -e
set -x

${APT_INSTALL} curl gpg
curl https://repo.anaconda.com/pkgs/misc/gpgkeys/anaconda.asc | gpg --dearmor > /usr/share/keyrings/conda-archive-keyring.gpg

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/conda-archive-keyring.gpg] https://repo.anaconda.com/pkgs/misc/debrepo/conda stable main" > /etc/apt/sources.list.d/conda.list

${APT_UPDATE}
${APT_INSTALL} conda
. /opt/conda/etc/profile.d/conda.sh
conda update -y conda
rm -r ~/.cache
mkdir ~/.cache
