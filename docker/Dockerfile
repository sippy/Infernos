# syntax=docker/dockerfile:1.7-labs

ARG BASE_IMAGE=ubuntu:24.10
FROM $BASE_IMAGE AS build
LABEL maintainer="Maksym Sobolyev <sobomax@sippysoft.com>"

USER root

# Build & install everything
WORKDIR /tmp
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt
ARG APT_UPDATE="apt-get update"
RUN ${APT_UPDATE}
ARG APT_UPGRADE="apt-get upgrade -y"
RUN ${APT_UPGRADE}
ARG APT_INSTALL="apt-get install --no-install-recommends -y"
RUN ${APT_INSTALL} lsb-release ca-certificates
COPY docker/install_conda.sh .
RUN ./install_conda.sh
COPY docker/setup_conda.sh .
ARG PYTHON_VER
ARG CONDA_MAINENV
ENV PYTHON_CMD="python${PYTHON_VER}"
RUN ./setup_conda.sh
COPY docker/install_hw.sh .
ARG INFER_HW
ENV CONDA_ACTIVATE="eval . /opt/conda/etc/profile.d/conda.sh && conda activate ${CONDA_MAINENV}"
RUN ./install_hw.sh
COPY docker/install_requirements.sh docker/intel-ray.diff requirements.txt .
ENV CONDA_MAINENV="${CONDA_MAINENV}"
RUN ./install_requirements.sh

COPY --exclude=.git --exclude=.github --link . /Infernos
WORKDIR /Infernos
