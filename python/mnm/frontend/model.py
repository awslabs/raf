from mnm._ffi.model import RunModel
from mnm.model.model import BaseModel
from mnm.model.trace import _unwrap


class FrameworkModel(BaseModel):
    def __init__(self, train_func, infer_func, params):
        super(FrameworkModel, self).__init__()
        self.__train_func = train_func
        self.__infer_func = infer_func
        self.__params = params

    def __call__(self, *args, **kwargs):
        func_inputs = [arg._ndarray__handle for arg in args]
        func_inputs += [param._ndarray__handle for param in self.__params.values()]
        if self._BaseModel__is_train:
            return _unwrap(RunModel(self.__train_func, func_inputs))
        return _unwrap(RunModel(self.__infer_func, func_inputs))

    def train_mode(self, recursive=True):
        self._BaseModel__is_train = True  # pylint: disable=invalid-name, attribute-defined-outside-init
        for param in self.__params.values():
            param.requires_grad = True

    def infer_mode(self, recursive=True):
        self._BaseModel__is_train = False  # pylint: disable=invalid-name, attribute-defined-outside-init
        for param in self.__params.values():
            param.requires_grad = False

    def get_relay_func(self, *args, **kwargs):
        ret = self.__train_func if self._BaseModel__is_train else self.__infer_func
        return ret

    def state(self, prefix="", recursive=True):
        return self.__params

    def to(self, *, ctx=None, dtype=None):  # pylint: disable=invalid-name
        for name, param in self.__params.items():
            new_param = param.to(ctx=ctx, dtype=dtype)
            self.__params[name] = new_param
