# pylint: line-too-long
"""Program build utilities."""
from ._ffi import build_info

def with_cuda():
    """Whether build with CUDA. if true, return the CUDNN version, or None otherwise."""
    if build_info.use_cuda():
        return build_info.cuda_version()
    return None

def git_version():
    """Current git version."""
    return build_info.git_version()

def with_cudnn():
    """Whether build with CUDNN. if true, return the CUDNN version, or None otherwise."""
    if build_info.use_cudnn() == "ON":
        return build_info.cudnn_version()
    return None

def with_mpi():
    """Whether MPI build is turned on."""
    return build_info.use_mpi() != "OFF"

def with_nccl():
    """Whether NCCL build is turned on."""
    return build_info.use_nccl() != "OFF"

def with_distributed():
    """Whether Distributed training is enabled."""
    return with_mpi() and with_nccl()

def with_cutlass():
    """Whether CUTLASS is enabled."""
    return build_info.use_cutlass() != "OFF"

def cmake_build_type():
    """Return cmake build type"""
    return build_info.cmake_build_type()
