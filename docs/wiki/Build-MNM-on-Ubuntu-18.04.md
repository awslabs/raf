This article introduces how to build MNM using CMake.

## Step 1. Install dependencies

**(Required) Build dependency**
<details>

```bash
sudo apt-get install ccache      # ccache is used to accelerate build
                     cmake       # hmm, cmake is required to run cmake
                     git

```

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

**(Optional) CUDA.** To run with latest CUDA, it is recommended to follow the instructions provided by NVIDIA, [link](https://developer.nvidia.com/cuda-downloads). The recommended setting is: Linux -> x86_64 -> Ubuntu -> 18.04 -> deb (network).

**(Optional) cuDNN.** NVIDIA provides cuDNN separately on its website, which requires additional account registration. Please follow the [link](https://developer.nvidia.com/rdp/cudnn-download), and following command yo decompress cuDNN.

<details>

```bash
tar zxvf cudnn-SOME-SUFFIX.tgz
```

</details>

## Step 2. Build MNM libraries

Below we introduce an environment variable that indicates where MNM is.

<details>

```bash
# Create the build directory
git clone https://github.com/meta-project/meta --recursive && cd meta
export MNM_HOME=$(pwd)
mkdir $MNM_HOME/build && cd $MNM_HOME/build
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

**Customize build.** By editing the configuration file `config.cmake`, one can easily customize the process of MNM build. Instructions are directly put inside the configuration file for convenience. For example, one may switch the cuDNN version by setting the `MNM_USE_CUDNN` or even by passing environment variables.

## Step 3. Run MNM

Here we come to the not-that-good part: to run MNM, one should properly set the environment variables.

<details>

```bash
export PYTHONPATH=$MNM_HOME/python/:$MNM_HOME/3rdparty/tvm/topi/python:$MNM_HOME/3rdparty/tvm/python
export TVM_LIBRARY_PATH=$MNM_HOME/build/lib
# The following commands can verify if the environments are set up correctly.
python -c "import mnm"
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
export MNM_HOME=PATH-TO-MNM
export PYTHONPATH=$MNM_HOME/python/:$MNM_HOME/3rdparty/tvm/topi/python:$MNM_HOME/3rdparty/tvm/python
export TVM_LIBRARY_PATH=$MNM_HOME/build/lib
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
