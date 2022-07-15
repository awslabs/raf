<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

This article introduces how to build RAF using CMake.

## Step 1. Install dependencies

If you don't want to spend time on setting up the environment, you could directly create a docker container with RAF docker images, which are mainly used by our CI. If you choose to do so, you could directly go to step 2.

```bash
# You could refer to gpu_image version in .github/workflows/ci_unit_test.yml
docker pull metaprojdev/raf:ci_gpu-v0.20
docker run --name work -it --gpus all metaprojdev/raf:ci_gpu-v0.20 /bin/bash
```

**(Required) Build dependency**
<details>

```bash
sudo apt-get install git ccache   # ccache is used to accelerate build
sudo snap install cmake --classic # hmm, cmake is required to run cmake
                                  # the cmake version installed by apt is too old
```

On the other hand, if you encounter any library missing errors during the compilation,
you could consider running the Ubuntu setup script here: `docker/install/ubuntu_install_core.sh`.
It makes sure all essential packages are installed.

Note that if you are using Ubuntu 20.10 or below, the ccache version via apt is 3.7-.
Since ccache 4.0- does not support nvcc (CUDA compiler) well, it will result in
cache miss for CUDA source files (e.g., CUTLASS). It means you may need to rebuild
ALL CUTLASS source files everytime. To resolve this issue, you can manually build
and install ccache 4.0+. Here are the steps of building ccache 4.0:

```bash
wget https://github.com/ccache/ccache/releases/download/v4.0/ccache-4.0.tar.gz
tar -xzf ccache-4.0.tar.gz
cd ccache-4.0
mkdir build; cd build
cmake -DZSTD_FROM_INTERNET=ON -DCMAKE_BUILD_TYPE=Release ..
make
sudo make install
```

It is recommended to build ccache 4.0 on Ubuntu 18, because later versions
require later glibc and other system libraries.

</details>

**(Optional) LLVM.** LLVM is required to enable operators written in TVM.

<details>

```bash
sudo apt-key adv --fetch-keys https://apt.llvm.org/llvm-snapshot.gpg.key
sudo apt-get update
sudo apt-get install libllvm-8-ocaml-dev libllvm8 llvm-8 llvm-8-dev           \
                     llvm-8-doc llvm-8-examples llvm-8-runtime                \
                     clang-8 clang-tools-8 clang-8-doc libclang-common-8-dev  \
                     libclang-8-dev libclang1-8 clang-format-10               \
                     python-clang-8 libfuzzer-8-dev lldb-8 lld-8              \
                     libc++-8-dev libc++abi-8-dev libomp-8-dev clang-tidy-8
```

</details>

**(Optional) CUDA.** RAF currently recommend to use CUDA 11.3.
It is recommended to follow the instructions provided by NVIDIA, [link](https://developer.nvidia.com/cuda-11.3.0-download-archive) with CUDA driver 465+.
The recommended setting is: Linux -> x86_64 -> Ubuntu -> 18.04 -> deb (network).
Then the CUDA paths need to be specified.
The following lines can be inserted to the `.bashrc` file for auto loading (see bonus below).

<details>

```bash
# this is for CUDA 11.3
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
```

</details>

**(Optional) cuDNN.** NVIDIA provides cuDNN separately on its website, which requires additional account registration. Please follow the [link](https://developer.nvidia.com/rdp/cudnn-download).
One can either download the tar ball file for Linux and use the following command to decompress cuDNN and specify the path, or download the `.deb` file to install. Please see the [link](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) for detailed instructions on installing cuDNN.

**(Optional) NCCL** NCCL is required for distributed training.
Like cuDNN, NVIDIA requires account registration to download NCCL. The detailed download and installation steps can be found from this [link](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#down).
A `.deb` file can be used for installation, specified by the CUDA version, the CPU architecture, and the OS.
The local installer allows local installation after downloading, and the network installer is much smaller but requires network connection for installation.

**(Optional) MPI** RAF uses MPI to do multi-process launch.
Any MPI implementation (e.g., OpenMPI, MPICH) is OK to serve this purpose.
The following command is used to install MPICH.

<details>

```bash
sudo apt-get install mpich
```

</details>

## Step 2. Build RAF libraries

Below we introduce an environment variable that indicates where RAF is.

<details>

```bash
# Create the build directory
git clone https://github.com/awslabs/raf --recursive && cd raf
export RAF_HOME=$(pwd)
mkdir $RAF_HOME/build
# Run the codegen for auto-generated source code
bash ./scripts/src_codegen/run_all.sh
# Configuration file for CMake
cd $RAF_HOME/build
cp ../cmake/config.cmake .
# Edit the configuration file
vim config.cmake
# Configure the project
cmake ..
# Finally let's trigger build
make -j$(nproc)
```

</details>

**Customize build.** By editing the configuration file `config.cmake`, one can easily customize the process of RAF build. Instructions are directly put inside the configuration file for convenience. For example, one may switch the cuDNN version by setting the `RAF_USE_CUDNN` or even by passing environment variables.

## Step 3. Install Python Dependecies

```
pip install six numpy pytest cython decorator scipy tornado typed_ast mypy orderedset pydot \
             antlr4-python3-runtime attrs requests Pillow packaging psutil dataclasses pycparser
```

Also, we recommend users to install PyTorch, see [Train PyTorch Modele](../2_user_guide/Train-PyTorch-Model.md).

## Step 4. Run RAF

Here we come to the not-that-good part: to run RAF, one should properly set the environment variables.
Again, the `export` commands can be put in `.bashrc` for auto loading (see bonus below).

<details>

```bash
export PYTHONPATH=$RAF_HOME/python/:$RAF_HOME/3rdparty/tvm/python
# The following commands can verify if the environments are set up correctly.
python3 -c "import raf"
```

</details>

## Bonus: Avoid setting environment variables every time

It is often annoying that environment variables are gone every time we open a new terminal, so sometimes we may want to set those variables globally, or inside a conda environment.

**Setting globally.** RC file is loaded automatically every time when a shell is set up. To determine which shell currently using, you may use "echo $SHELL".

<details>

```bash
# If using bash
vim $HOME/.bashrc
# If using zsh
vim $HOME/.zshrc
# Adding the export commands to the end of those RC files
export RAF_HOME=PATH-TO-RAF
export PYTHONPATH=$RAF_HOME/python/:$RAF_HOME/3rdparty/tvm/python
```

</details>

**Set up in conda environment.** If you are concerned that the solution above may make the RC files cumbersome, another possible way is that we can set environment variables in conda's activation script.

<details>

```bash
# First, enter your conda environment
conda activate your-conda-env
# Put export commands into this file
mkdir -p $CONDA_PREFIX/etc/conda/activate.d/
vim $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

</details>
