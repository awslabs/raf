<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

(Document work-in-progress)

This article introduces the way of using [conda](https://conda.io/) as working environment, as it provides fine-grained version control for rich amount of packages, without asking for root permission.

## Install dependencies

**(Required) Build dependency**

<details>

```bash
# Create an environment called raf-dev
conda create -n raf-dev python=3.7 cmake cudnn
# Entering the environment
conda activate raf-dev
# Installing necessary packages
conda install clangdev=8.0.1 ccache=4.3 ipdb -c conda-forge
# Then, verify if llvm are correctly installed
which llvm-config
```

Note that for certain dependencies (e.g., ccache), the compiler version (e.g., GCC) that the dependency requires may mismatch with the system default version (e.g. GCC 7.5.0 for Ubuntu 18.04) and Conda wants to install a new compiler. It is **NOT** recommended to proceed with installation immediately if the version of the new compiler vastly differs from the default one. Instead, we could try to install the dependencies one by one, figure out the one causes this issue, and manually select (usually downgrade to) a proper version of this dependency to install.

</details>

**(Optional) NCCL.** NCCL is required for distributed training.

<details>

```bash
conda install nccl -c conda-forge
```

However, there is a issue that CMake won't be aware of NCCL Library newly installed by Conda. Thus, we need to hint CMake by setting the following environment variable.

```bash
export NCCL_DIR=$CONDA_PREFIX
```

</details>

**(Optional) MPI.** MPI is used to launch distributed RAF on multiple nodes, and assist NCCL to initialize communicators. 

<details>

```bash
conda install openmpi -c conda-forge
```

Note that installing MPI via Conda is **NOT** a common option, and we found that sometimes it can be troublesome. If there is a MPI library provided by,

- **(Recommended for regular users)** System package managers (e.g., apt)
- Hardware vendors (e.g. OpenMPI from NVIDIA HPC SDK)
- System administrators (typically on supercomputers)
- [Spack](https://spack.io/), a package manager for supercomputers without asking for root permission

then it is suggested to use them first rather than install a new one through Conda.

</details>

## Persist environment variables

Conda also provides a activation script that is run every time we enter the environment:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d/
vim $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

We may set environment variables inside this script:

```bash
export CC=$(which gcc)
export CXX=$(which g++)
export CUDA_HOME=/usr/local/cuda 
export RAF_HOME=$HOME/Projects/raf-dev
export TVM_HOME=$RAF_HOME/3rdparty/tvm
export PYTHONPATH=$RAF_HOME/python/:$TVM_HOME/python
export CUDNN_HOME=$CONDA_PREFIX
export NCCL_DIR=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "CC = $CC"
echo "CXX = $CXX"
echo "CUDA_HOME = $CUDA_HOME"
echo "RAF_HOME = $RAF_HOME"
echo "TVM_HOME = $TVM_HOME"
echo "PYTHONPATH = $PYTHONPATH"
echo "CUDNN_HOME = $CUDNN_HOME"
echo "NCCL_DIR = $NCCL_DIR"
```
