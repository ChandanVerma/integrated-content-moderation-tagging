# FROM nvidia/cuda:11.1.1-base-ubuntu20.04
# RUN apt update
# RUN apt install -y tzdata
# RUN apt update -y && \
#     apt install -y apt-utils && \
#     apt install -y build-essential software-properties-common gcc git && \
#     add-apt-repository -y ppa:deadsnakes/ppa && \
#     apt-get install ffmpeg libsm6 libxext6  -y && \
#     apt install -y python3.8 python3-distutils python3-pip python3-setuptools wget && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*
# RUN ln -s /usr/bin/python3 /usr/bin/python
# WORKDIR /app
# COPY . /app
# RUN pip install -r /app/requirements.txt

FROM rayproject/ray:1.11.1-gpu
USER root
COPY system-requirements.txt /app/
RUN rm /etc/apt/sources.list.d/nvidia-ml.list \
  && apt-get clean \
  && sudo apt-key del 7fa2af80 \
  && sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN sudo apt-get update --fix-missing \
  && sudo apt-get -y install git \
  && sudo apt-get install -y $(cat /app/system-requirements.txt | tr "\n" " ") \
  && sudo apt-get clean \
  && sudo rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app/
# RUN sudo mv ./cache_models/* ~/.cache/ \
# && sudo rm -rf ./cache_models/
RUN pip install -r requirements.txt \
  && pip install git+https://github.com/openai/CLIP.git --no-deps
  