"""Model that wraps the IR converted from deep learning frameworks."""
from mnm._ffi.model import RunModel
from mnm.model.model import BaseModel
from mnm.model.trace import _unwrap


class FrameworkModel(BaseModel):
    """Represent the wrapper of the models read from deep learning frameworks.

    Parameters
    ----------
    train_func : Expr
        Function contains both forward and backward computation.

    infer_func : Expr
        Forward function

    arg_params : Dict[str, ndarray]
        Model parameters

    aux_params : Dict[str, ndarray]
        Auxiliary params, not learnable
    """
    def __init__(self, train_func, infer_func, arg_params, aux_params):
        super(FrameworkModel, self).__init__()
        self.__train_func = train_func
        self.__infer_func = infer_func
        self.__arg_params = arg_params
        self.__aux_params = aux_params
        for param in self.__aux_params.values():
            param.requires_grad = False

    def __call__(self, *args, **kwargs):
        func = self.__infer_func
        if self._BaseModel__is_train:
            func = self.__train_func
        func_inputs = list()
        arg_index = 0
        for var_node in func.params:
            var_name = var_node.name_hint
            if var_name in self.__arg_params:
                func_inputs.append(self.__arg_params[var_name]._ndarray__handle)
            elif var_name in self.__aux_params:
                func_inputs.append(self.__aux_params[var_name]._ndarray__handle)
            elif var_name in kwargs:
                func_inputs.append(kwargs[var_name]._ndarray__handle)
            else:
                func_inputs.append(args[arg_index]._ndarray__handle)
                arg_index += 1
        return _unwrap(RunModel(func, func_inputs))

    def train_mode(self, recursive=True):
        self._BaseModel__is_train = True  # pylint: disable=invalid-name, attribute-defined-outside-init
        for param in self.__arg_params.values():
            param.requires_grad = True

    def infer_mode(self, recursive=True):
        self._BaseModel__is_train = False  # pylint: disable=invalid-name, attribute-defined-outside-init
        for param in self.__arg_params.values():
            param.requires_grad = False

    def get_relay_func(self, *args, **kwargs):
        ret = self.__train_func if self._BaseModel__is_train else self.__infer_func
        return ret

    def state(self, prefix="", recursive=True):
        return self.__arg_params

    def to(self, *, ctx=None, dtype=None):  # pylint: disable=invalid-name
        for name, param in self.__arg_params.items():
            new_param = param.to(ctx=ctx, dtype=dtype)
            self.__arg_params[name] = new_param
        for name, param in self.__aux_params.items():
            new_param = param.to(ctx=ctx, dtype=dtype)
            self.__aux_params[name] = new_param
