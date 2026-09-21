FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy source code (assumes you're building from repo root)
COPY . /workspace

# Do NOT install Python deps here
# Avoid partial environments

CMD ["/bin/bash"]


















# below dockerfile gives error when runnign lerobot-record command
# ImportError: /opt/conda/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda9SetDeviceEi


# FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# ENV DEBIAN_FRONTEND=noninteractive

# RUN apt-get update && apt-get install -y \
#     git \
#     curl \
#     wget \
#     build-essential \
#     cmake \
#     ninja-build \
#     usbutils \
#     libgl1 \
#     libglib2.0-0 \
#     libusb-1.0-0-dev \
#     udev \
#     ffmpeg \
#     && rm -rf /var/lib/apt/lists/*



# RUN apt-get update && apt-get install -y \
#     ffmpeg \
#     libx264-dev \
#     && rm -rf /var/lib/apt/lists/*


# # Python tools
# RUN pip install --upgrade pip setuptools wheel
# RUN pip install ninja
# RUN pip install --no-build-isolation flash-attn



# # Install LeRobot (and dependencies)
# RUN pip install \
#     opencv-python \
#     pyrealsense2 \
#     scipy \
#     transformers \
#     accelerate

# # via docs, must do custom install like this:
# RUN pip install lerobot[groot]

# # for some reason lerobot does not autoinstall this
# RUN pip install feetech-servo-sdk==1.0.0

# # lerobot docs, conda included in docker image : pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
# RUN conda install -y -c conda-forge ffmpeg=7.1.1

# # lerobot-record uses text to speech
# RUN apt-get update && apt-get install -y speech-dispatcher


# # Optional: debugging tools
# RUN pip install ipython

# WORKDIR /workspace