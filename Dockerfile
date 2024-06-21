FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel


#RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update && apt-get install -y \
    git \
    clang-10 \
    libc++-10-dev \
    libc++abi-10-dev \
    wget \
    ninja-build \
    python3-pip \
    libpng-dev \
    libjpeg-dev \
    libpython3-dev \
    build-essential \
    libssl-dev \
    llvm-10-dev \
    libopenexr-dev \
    openssl \
    cmake \
    python3-distutils && \
    rm -rf /var/lib/apt/lists/*


# Mitsuba
RUN pip3 install \
    mitsuba \
    drjit


WORKDIR /codes



#COPY codes /codes/
#WORKDIR /codes/build
#RUN cmake -DCMAKE_C_COMPILER=clang-10 -DCMAKE_CXX_COMPILER=clang++-10 -GNinja .. 
#RUN ninja
#RUN source setpath.sh

