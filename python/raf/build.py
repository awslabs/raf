# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Program build utilities."""
from ._ffi import build_info


def git_version():
    """Current git version."""
    return build_info.git_version()


def with_cuda():
    """Whether build with CUDA. if true, return the CUDA version, or None otherwise."""
    if build_info.use_cuda():
        return build_info.cuda_version()
    return None


def with_cublas():
    """Whether build with CUBLAS."""
    return build_info.use_cublas() != "OFF"


def with_cudnn():
    """Whether build with CUDNN. if true, return the CUDNN version, or None otherwise."""
    if build_info.use_cudnn() != "OFF":
        return build_info.cudnn_version()
    return None


def with_mpi():
    """Whether MPI build is turned on."""
    return build_info.use_mpi() != "OFF"


def with_nccl():
    """Whether build with NCCL, if true, return NCCL version, or None otherwise."""
    if build_info.use_nccl():
        return build_info.nccl_version()
    return None


def with_distributed():
    """Whether Distributed training is enabled."""
    return with_mpi() and with_nccl()


def with_cutlass():
    """Whether CUTLASS is enabled."""
    return build_info.use_cutlass() != "OFF"


def cmake_build_type():
    """Return cmake build type"""
    return build_info.cmake_build_type()


def build_with(backend):  # pylint: disable=too-many-return-statements
    """Check if a backend is built with RAF.

    Parameters
    ----------
    backend : str
        The backend name

    Returns
    -------
    Whether the backend is built with RAF.
    """
    assert backend in ["tvm", "cuda", "cudnn", "cutlass", "cublas", "nccl"], (
        "Invalid backend: %s" % backend
    )
    if backend == "tvm":
        return True  # it seems like that we always build with TVM
    if backend == "cuda":
        return with_cuda() is not None
    if backend == "cublas":
        return with_cublas()
    if backend == "cudnn":
        return with_cudnn() is not None
    if backend == "cutlass":
        return with_cutlass()
    if backend == "nccl":
        return with_nccl() is not None
    return False
