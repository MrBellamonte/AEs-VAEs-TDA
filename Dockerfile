#FROM ubuntu:14.04
##MAINTAINER "Joshua C. Randall" <jcrandall@alum.mit.edu>
#
## Prerequisites
#RUN \
#  apt-get update && \
#  apt-get -y upgrade && \
#  apt-get install -y python3.7 python3-pip python-dev
#
#
#In case someone else is ok with getting Python3.6 installed as a side effect (python3.7-distutils introduces it as pointed out by OP). This will install Python3.7 making it default and have the latest available pip using your python3.7 installation

#FROM ubuntu:18.04
FROM ubuntu:18.04

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 10.2.89
ENV CUDA_PKG_VERSION 10-2=$CUDA_VERSION-1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-$CUDA_PKG_VERSION \
    cuda-compat-10-2 \
    && ln -s cuda-10.2 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.2 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441"

# (...)

# Python package management and basic dependencies
#RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

## Register the version in alternatives
#RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
#
## Set python 3 as the default python
#RUN update-alternatives --set python /usr/bin/python3.7
#
## Upgrade pip to latest version
#RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
#    python get-pip.py --force-reinstall && \
#    rm get-pip.py
# install python, ninja and git
#RUN apt-get update -y
#RUN apt-get install python3.7 -y
#RUN apt-get install python3-pip -y
#RUN pip install -U pip
#RUN apt-get install nvidia-container-runtime -y
#RUN apt-get install build-essential autoconf libtool pkg-config python-opengl python-imaging python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev -y
RUN apt-get install build-essential -y
#RUN apt-get install gcc -y
RUN apt-get install python-dev gcc -y

RUN pip install Cython
RUN pip install numpy
RUN pip install cmake
RUN pip install  pybind11
# install requirements
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# other requirements
RUN pip install dataclasses
RUN pip install torch torchvision
RUN pip install numpy
RUN pip install scikit-learn --upgrade
RUN pip install umap
RUN pip install umap-learn

RUN apt-get install ninja-build -y
RUN apt-get install -y git

RUN git clone https://github.com/MrBellamonte/torchph.git

RUN cd torchph && python setup.py install


COPY . /MT-VAEs-TDA
WORKDIR "MT-VAEs-TDA"
#
#RUN pwd

