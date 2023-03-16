# Installation Instruction

Note 1: though our project supports ROS integration, it is not included in the current installation instruction.

Note 2: for convenience, the installation instruction we wrote assumes installation to `/usr/local` with root access.
In reality, to avoid collision with existing system packages, it is
recommended to install all packages (including the ones that OpenVSLAM depends on) into
a local directory such as `~/.local` instead of the system path `/usr/local`. This can
be easily achieved by adding `-DCMAKE_INSTALL_PREFIX=$HOME/.local` option to a CMake command.
Another benifit of this is that `sudo` privilege is not needed for the installation, which may be desirable on multi-user servers.

## Dependancies
* Build System
    * C++14
    * CMake >= 3.18
    * CUDA >= 11.0
* Externel Libraries
    * A previous build of [OpenVSLAM](https://arxiv.org/pdf/1910.01122.pdf). The project has not been tested on the newest version of OpenVSLAM and depends on a custom build of OpenVSLAM, which we provide [here](https://drive.google.com/file/d/1DayZLNd8hTMVM02SGUwFVS30Jlv-Zjuc/view?usp=sharing). We have to use this custom build to expose certain critical private variables for our usage. However, **NOTE THAT THIS BUILD OF OPENVSLAM CARRIES AN INCORRECT LICENSE** (details [here](https://github.com/OpenVSLAM-Community/openvslam/issues/249)). Therefore, when building the OpenVSLAM library, please change the license to GPL license.
    * [LibRealSense](https://github.com/IntelRealSense/librealsense)

Please follow links provided above for instructions to install CUDA and LibRealSense. We provide detailed instruction to install CMake and OpenVSLAM here.

We assume that your user name is URUSERNAME and you are building dependencies in `/home/URUSERNAME/dep` and dependencies are installed system-wide (i.e., root access).

```
cd ~/
mkdir dep
cd dep
```

## General Dependencies

First, install some general dependencies (if you have not gotten these on your computer)

```
apt update
apt install -y build-essential pkg-config cmake git wget curl unzip
# g2o dependencies
apt install -y libatlas-base-dev libsuitesparse-dev
# OpenCV dependencies
apt install -y libgtk-3-dev
apt install -y ffmpeg
apt install -y libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavresample-dev
# eigen dependencies
apt install -y gfortran
# backward-cpp dependencies (optional)
apt install -y binutils-dev
# other dependencies
apt install -y libyaml-cpp-dev libgflags-dev

# Pangolin dependencies
apt install -y libglew-dev
```

Now, install a newer version of CMake. Here we use 3.22.3 as an example:

```
wget https://github.com/Kitware/CMake/releases/download/v3.22.3/cmake-3.22.3.tar.gz
tar xvf cmake-3.22.3.tar.gz
./configure
make -j4
sudo make install
```

Try running `cmake --version` to check if it has the updated version. Sometimes make install does not overwrite the old cmake depending on how you installed your old cmake. If the version is not updated (or it throws an error), you may need to update the `$PATH` variable by writing the following line to your `~/.bashrc` or `~/.zshrc` file:

```
export PATH=/home/URUSERNAME/dep/cmake-3.22.3/bin:$PATH
source ~/.bashrc # or ~/.zshrc for ZSH users
```

## Dependencies for OpenVSLAM

Before installing the OpenVSLAM library, you need to install some dependencies for it. The official instruction is available [here](https://openvslam-community.readthedocs.io/en/latest/installation.html), but since OpenVSLAM is a library under active development, the installation instruction may fail for the custom build we use (known to happen in the past). So here we provide a detailed instruction.

Download and install Eigen from source

```
cd ~/dep
wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xvf eigen-3.4.0.tar.gz
rm eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir -p build && cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    ..
make -j4
sudo make install
```

Download and build OpenCV with some custom options. Here we use OpenCV 4.5.5 as an example. In addition, we recommend installing OpenCV with OpenCV Contrib module, which includes implementation of many recent visual algorithms. Though the project is not using OpenCV contrib now, we are exploring options in OpenCV contrib to further improve our system.

```
cd ~/dep
wget https://github.com/opencv/opencv/archive/4.5.5.zip
unzip 4.5.5.zip # inflate to ./opencv-4.5.5/
rm 4.5.5.zip
wget https://github.com/opencv/opencv_contrib/archive/refs/tags/4.5.5.zip
unzip 4.5.5.zip # inflate to ./opencv_contrib-4.5.5/
rm 4.5.5.zip
cd opencv-4.5.5
mkdir build && cd build
cmake \                            
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DENABLE_CXX11=ON \
    -DBUILD_DOCS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_JASPER=OFF \
    -DBUILD_OPENEXR=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_TESTS=OFF \
    -DWITH_EIGEN=ON \
    -DWITH_FFMPEG=ON \
    -DWITH_OPENMP=OFF -DINSTALL_PYTHON_EXAMPLES=ON -DBUILD_OPENCV_PYTHON3=ON -DOPENCV_EXTRA_MODULES_PATH=/home/URUSERNAME/dep/opencv_contrib-4.5.5/modules -DWITH_CUDA=ON -DENABLE_FAST_MATH=1 -DCUDA_FAST_MATH=1 -DWITH_CUBLAS=1 \
    ..\
make -j4 # this takes a while...
sudo make install
```

Download a custom build of DBoW2 from [here](https://drive.google.com/file/d/1TV2OAwQyMlPZlIfFAE7Jfop27F81eOKS/view?usp=sharing).

```
unzip DBoW2.zip
cd DBoW2
mkdir build && cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    ..
make -j4
sudo make install
```

Download, build and install g2o.

```
cd ~/dep
git clone https://github.com/RainerKuemmerle/g2o.git
cd g2o
git checkout 9b41a4ea5ade8e1250b9c1b279f3a9c098811b5a
mkdir build && cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_CXX_FLAGS=-std=c++11 \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_UNITTESTS=OFF \
    -DG2O_USE_CHOLMOD=OFF \
    -DG2O_USE_CSPARSE=ON \
    -DG2O_USE_OPENGL=OFF \
    -DG2O_USE_OPENMP=ON \
    ..
make -j4
sudo make install
```

Download, build and install Pangolin Viewer from source.

```
cd ~/dep
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
git checkout ad8b5f83222291c51b4800d5a5873b0e90a0cf81
mkdir build && cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    ..
make -j4
sudo make install
```

## Installing OpenVSLAM

Now you are ready to install the OpenVSLAM library. First obtain the custom build we provide from the link above, put it in `~/dep`, and run,

```
unzip openvslam-system_inherit.zip
cd openvslam-system_inherit
mkdir build && cd build
cmake \
    -DUSE_PANGOLIN_VIEWER=ON \
    -DINSTALL_PANGOLIN_VIEWER=ON \
    -DUSE_SOCKET_PUBLISHER=OFF \
    -DBUILD_TESTS=ON \
    -DBUILD_EXAMPLES=ON \
    ..
make -j4
sudo make install
```

## Installing PyTorch

At the time of writing this doc, the repo is using libtorch (C++ version of PyTorch) for inference.
We provide instructions for X86/64 (PC) and Nvidia Jetson (ARM) users.

### X86/64 users

Go to the [PyTorch installation page](https://pytorch.org/get-started/locally/) to install a CUDA-enabled PyTorch first.
Then, on the same page, click C++/Java and download pre-built libtorch release. Note that there are two versions of libtorch:
one with pre-cxx11 ABI and one with cxx11 ABI. Download the one with cxx11 ABI and unzip to the `third_party/` folder
under project folder.

```
cd ~/dep
wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu102.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cu102.zip
cd ~/RA-SLAM/third_party
ln -s ~/dep/libtorch libtorch
```

The CMake file is hinted to look for libtorch under third_party.

### Nvidia Jetson TX2/XAVIER

We assume that Jetson users use images provided by Nvidia official with Jetpack.
In this case, pre-built PyTorch and libtorch packages are available at [this forum page](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-10-now-available/72048).
Installing the pre-built wheel will install **both** PyTorch and libtorch to the default python package folder.
We hint the CMake file to look for commonly used python3 installation path, but in case that fails you can modify the CMake file and point the hint to correct path.

## Building RA-SLAM

Now you are finally ready to build RA-SLAM! Good job getting here.

```
mkdir build && cd build
cmake ..
make -j4
```

## Testing examples

Check back [README.md](./README.md) for examples!
