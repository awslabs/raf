"""Program build utilities."""
from ._ffi import build_info


def with_cuda():
    """Whether CUDA build is turned on."""
    return build_info.use_cuda() != "OFF"

def git_version():
    """Current git version."""
    return build_info.git_version()
