"""Program build utilities."""
from ._ffi import build_info


def with_cuda():
    """Whether CUDA build is turned on."""
    return build_info.use_cuda()

def git_version():
    """Current git version."""
    return build_info.git_version()

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
