#!/bin/bash
set -e 

sudo ln -sf /usr/bin/python3 /usr/bin/python

# pip
sudo apt-get update
sudo apt-get install -y python3-pip

# Jtop
sudo -H pip install -U jetson-stats

# Set up CUDA environment
if [ ! -x "$(command -v nvcc)" ]; then
    echo "export PATH=/usr/local/cuda/bin\${PATH:+:\${PATH}}" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" >> ~/.bashrc
    # Trick
    export PATH=/usr/local/cuda/bin\${PATH:+:\${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}
fi

# NumPy
sudo apt-get install -y python3-pip libhdf5-serial-dev hdf5-tools libcanberra-gtk-module
sudo -H pip3 install Cython
sudo pip3 -H install numpy==1.19

# SciPy and Sklearn
sudo apt-get install -y libatlas-base-dev gfortran
sudo apt-get install -y libpcap-dev libpq-dev
sudo -H pip3 install scikit-learn

# Numba
sudo apt-get install -y llvm-8 llvm-8-dev
sudo -H LLVM_CONFIG=/usr/bin/llvm-config-8 pip3 install numba==0.48

# CuPy
echo "Installing CuPy, this may take a while..."
sudo -H CUPY_NVCC_GENERATE_CODE="current" CUPY_NUM_BUILD_JOBS=$(nproc) pip3 install cupy==9.2

# end
echo " ===================================="
echo " |       REBOOT IS REQUIRED         |"
echo " ===================================="

