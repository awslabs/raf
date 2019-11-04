from mnm._lib import _DLContext
from mnm._lib import _NodeBase as NodeBase
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


def _get_ctx_map():
    DEV_TYPE_MASK = {
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
    _ctx2str = {}

    for device_type, idx in DEV_TYPE_MASK.items():
        _str2ctx[device_type] = _DLContext(device_type=idx, device_id=0)

        for device_id in range(128):
            _str2ctx[f"{device_type}({device_id})"] = _DLContext(device_type=idx, device_id=device_id)
            _ctx2str[(idx, device_id)] = f"{device_type}({device_id})"

    return _str2ctx, _ctx2str


_STR2CTX, _CTX2STR = _get_ctx_map()


def ctx2str(ctx: _DLContext) -> str:
    return _CTX2STR[(int(ctx.device_type), int(ctx.device_id))]


def str2ctx(name: str) -> _DLContext:
    return _STR2CTX[name]
