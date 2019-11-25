from mnm._lib import _DLContext
from mnm._lib import _NodeBase as NodeBase  # pylint: disable=unused-import
from mnm._lib import _register_node


def register_node(type_key=None):
    assert isinstance(type_key, str)

    return _register_node(type_key)


def set_module(module):
    def decorator(func):
        if module is not None:
            func.__module__ = module

        return func

    return decorator


def get_func_name(pyfunc):
    return pyfunc.__module__ + "$" + pyfunc.__qualname__


def _get_ctx_map():
    dev_type_mask = {
        'llvm': 1,
        'stackvm': 1,
        'cpu': 1,
        'c': 1,
        'gpu': 2,
        'cuda': 2,
        'nvptx': 2,
        'cl': 4,
        'opencl': 4,
        'aocl': 5,
        'aocl_sw_emu': 5,
        'sdaccel': 6,
        'vulkan': 7,
        'metal': 8,
        'vpi': 9,
        'rocm': 10,
        'opengl': 11,
        'ext_dev': 12,
        'micro_dev': 13,
    }
    _str2ctx = {}

    for device_type, idx in dev_type_mask.items():
        _str2ctx[device_type] = _DLContext(device_type=idx, device_id=0)

        for device_id in range(128):
            name = f"{device_type}({device_id})"
            _str2ctx[name] = _DLContext(device_type=idx, device_id=device_id)

    return _str2ctx


_STR2CTX = _get_ctx_map()


def ctx2str(ctx: _DLContext) -> str:
    mask = [
        None,
        "cpu",
        "cuda",
        "cpu_pinned",
        "cl",
        "aocl",
        'sdaccel',
        'vulkan',
        'metal',
        'vpi',
        'rocm',
        'opengl'
    ]
    dev_type = int(ctx.device_type)
    dev_id = int(ctx.device_id)

    if dev_id == 0 and dev_type in (1, 3):
        return mask[dev_type]

    return mask[dev_type] + "(" + str(dev_id) + ")"


def str2ctx(name: str) -> _DLContext:
    return _STR2CTX[name]
