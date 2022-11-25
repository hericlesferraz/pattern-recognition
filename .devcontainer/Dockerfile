FROM nvidia/cuda:11.8.0-base-ubuntu22.04

ARG USERNAME=dev_machine
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

WORKDIR /home/$USERNAME

RUN apt-get update \
    && apt-get install -y sudo \
    && sudo apt install  software-properties-common -y \
    && sudo apt install -y python3-pip \
    && sudo apt install ffmpeg -y libsm6 libxext6 \
    && sudo apt install git -y

#COPY ~/.gitconfig /home/$USERNAME/.gitconfig