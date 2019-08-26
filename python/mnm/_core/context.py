from .._ffi._tvm import _DLContext
from .base import set_module


def __init_name_dict():
    STR2MASK = {
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
    name_dict = {}
    for device_type, idx in STR2MASK.items():
        name_dict[device_type] = (idx, 0)
        for device_id in range(128):
            name_dict[f"{device_type}({device_id})"] = (idx, device_id)
    return name_dict


_NAME_DICT = __init_name_dict()


@set_module("mnm")
class Context(_DLContext):

    def __init__(self, name: str):
        if isinstance(name, str):
            result = _NAME_DICT.get(name, None)
            if result is None:
                raise ValueError("Cannot recognize device context: " + name)
            device_type, device_id = result
            self.device_type = device_type
            self.device_id = device_id
        elif isinstance(name, _DLContext):
            self.device_type = name.device_type
            self.device_id = name.device_id
        else:
            raise NotImplementedError(name)

    @staticmethod
    def create(device_type, device_id):
        return Context(_DLContext(device_type, device_id))


@set_module("mnm")
def cpu(dev_id=0):
    return Context.create(1, dev_id)


@set_module("mnm")
def gpu(dev_id=0):
    return Context.create(2, dev_id)
