from ._ffi import build_info


def with_cuda():
    return build_info.use_cuda() != "OFF"

def git_version():
    return build_info.git_version()
