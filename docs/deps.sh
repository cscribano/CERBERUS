sudo apt-get install -y libhdf5-serial-dev hdf5-tools libcanberra-gtk-module
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
