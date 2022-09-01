<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

Author: [Zihao Ye](https://github.com/yzh119/)

This article introduces how to build RAF using CMake on macOS.

## Step 1. Install dependencies

**(Required) Build dependency** 
<details>

```bash
brew install ccache      # ccache is used to accelerate build
             cmake       # cmake is required to run cmake
             git
```

</details>

**(Optional) LLVM.** LLVM is required to enable operators written in TVM.


<details>

```bash
brew install llvm
```

</details>

## Step 2. Build RAF libraries

Below we introduce an environment variable that indicates where RAF is.

<details>

```bash
# Create the build directory
git clone https://github.com/awslabs/raf --recursive && cd raf
export RAF_HOME=$(pwd)
mkdir $RAF_HOME/build && cd $RAF_HOME/build
# Configuration file for CMake
cp ../cmake/config.cmake .
# Edit the configuration file
vim config.cmake
# Configure the project
cmake ..
# Finally let's trigger build
make -j$(nproc)
```

</details>

**Customize build.** By editing the configuration file `config.cmake`, one can easily customize the process of RAF build. Instructions are directly put inside the configuration file for convenience. 

## Step 3. Run RAF

Here we come to the not-that-good part: to run RAF, one should properly set the environment variables.

<details>

```bash
export PYTHONPATH=$RAF_HOME/python/:$RAF_HOME/3rdparty/tvm/topi/python:$RAF_HOME/3rdparty/tvm/python
# The following commands can verify if the environments are set up correctly.
python -c "import raf"
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
export PYTHONPATH=$RAF_HOME/python/:$RAF_HOME/3rdparty/tvm/topi/python:$RAF_HOME/3rdparty/tvm/python
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
