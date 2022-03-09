# main image
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# tweaked azureml pytorch image
# as of Aug 31, 2020 pt 1.6 doesn't seem to work with horovod on native mixed precision

LABEL maintainer="Albert"
LABEL maintainer_email="alsadovn@microsoft.com"
LABEL version="0.1"

USER root:root

ENV com.nvidia.cuda.version $CUDA_VERSION
ENV com.nvidia.volumes.needed nvidia_driver
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV NCCL_DEBUG=INFO
ENV HOROVOD_GPU_ALLREDUCE=NCCL

# Install Common Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # SSH and RDMA
    libmlx4-1 \
    libmlx5-1 \
    librdmacm1 \
    libibverbs1 \
    libmthca1 \
    libdapl2 \
    dapl2-utils \
    openssh-client \
    openssh-server \
    iproute2 && \
    # Others
    apt-get install -y --no-install-recommends \
    build-essential \
    bzip2=1.0.6-8.1ubuntu0.2 \
    libbz2-1.0=1.0.6-8.1ubuntu0.2 \
    systemd \
    git=1:2.17.1-1ubuntu0.7 \
    wget \
    cpio \
    libsm6 \
    libxext6 \
    libxrender-dev \
    fuse && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Conda Environment
ENV MINICONDA_VERSION 4.7.12.1
ENV PATH /opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda clean -ay && \
    rm -rf /opt/miniconda/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

# To resolve horovod hangs due to a known NCCL issue in version 2.4.
# Can remove it once we upgrade NCCL to 2.5+.
# https://github.com/horovod/horovod/issues/893
# ENV NCCL_TREE_THRESHOLD=0
ENV PIP="pip install --no-cache-dir"

RUN conda install -y conda=4.8.5 python=3.6.2 && conda clean -ay && \
    conda install -y mkl=2020.1 && \
    conda install -y numpy scipy scikit-learn scikit-image imageio protobuf && \
    conda install -y ruamel.yaml==0.16.10 && \
    # ruamel_yaml is a copy of ruamel.yaml package
    # conda installs version ruamel_yaml v0.15.87 which is vulnerable
    # force uninstall it leaving other packages intact
    conda remove --force -y ruamel_yaml && \
    conda clean -ay && \
    # Install AzureML SDK
    ${PIP} azureml-defaults && \
    # Install PyTorch
    ${PIP} torch==1.4.0 && \
    ${PIP} torchvision==0.2.1 && \
    ${PIP} wandb && \
    # # Install Horovod
    # HOROVOD_WITH_PYTORCH=1 ${PIP} horovod[pytorch]==0.19.5 && \
    # ldconfig && \
    ${PIP} tensorboard==1.15.0 && \
    ${PIP} future==0.17.1 && \
    ${PIP} onnxruntime==1.4.0 && \
    ${PIP} pytorch-lightning && \
    ${PIP} opencv-python-headless~=4.4.0 && \
    ${PIP} imgaug==0.4.0 --no-deps && \
    # hydra
    ${PIP} hydra-core --upgrade && \
    ${PIP} lmdb pyarrow

RUN pip3 install --upgrade pip
RUN pip3 install pipreqs

RUN apt-get update
RUN apt-get install -y --no-install-recommends libglib2.0-dev
RUN apt-get install -y --no-install-recommends vim           

WORKDIR /
RUN apt-get install -y --no-install-recommends libunwind8
RUN apt-get install -y --no-install-recommends libicu-dev
RUN apt-get install -y --no-install-recommends htop
RUN apt-get install -y --no-install-recommends net-tools
RUN apt-get install -y --no-install-recommends rsync
RUN apt-get install -y --no-install-recommends tree

RUN  wget -O azcopy.tar.gz https://aka.ms/downloadazcopylinux64
RUN tar -xf azcopy.tar.gz
RUN ./install.sh

# put the requirements file for your own repo under /app for pip-based installation!!!
WORKDIR /app
RUN pip3 install -r requirements.txt
