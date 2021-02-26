"""Model that wraps the IR converted from deep learning frameworks."""
# pylint: disable=protected-access, attribute-defined-outside-init
from tvm import relay

from mnm._ffi.model import RunModel
from mnm.model.model import BaseModel
from mnm.model.trace import _unwrap, _TraceRecord
from mnm._core.ndarray import ndarray, Symbol
from mnm._ffi.pass_ import AssignDevice, Substitute


def _get_func_inputs(model, args, kwargs, get_handle=True):
    arg_index = 0
    res = []
    func = model._FrameworkModel__train_func \
        if model._BaseModel__is_train else model._FrameworkModel__infer_func
    for var_node in func.params:
        var_name = var_node.name_hint
        if var_name in model._FrameworkModel__arg_params:
            res.append(model._FrameworkModel__arg_params[var_name])
        elif var_name in model._FrameworkModel__aux_params:
            res.append(model._FrameworkModel__aux_params[var_name])
        elif var_name in kwargs:
            res.append(kwargs[var_name])
        else:
            res.append(args[arg_index])
            arg_index += 1
    if get_handle:
        res = [i._ndarray__handle for i in res]
    return res


def annotate_func_params(func, args):
    """Annotate func parameters with the type of args

    Parameters
    ----------
    func : relay.Function
        the function to be annotated

    args : list[ndarray]
        function input arguments

    Returns
    -------
    return : relay.Function
        the annotated function
    """
    def get_type(arg):
        if isinstance(arg, ndarray):
            return relay.TensorType(shape=arg.shape, dtype=arg.dtype)
        if isinstance(arg, Symbol):
            return arg._Symbol__handle.type_annotation
        raise TypeError("Not supported: ", type(arg))
    params = [relay.Var(param.name_hint, get_type(arg)) for param, arg in zip(func.params, args)]
    vmap = dict(zip(func.params, params))
    body = Substitute(func.body, vmap)
    return relay.Function(params, body)


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
    # pylint: disable=invalid-name

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
        func_inputs = _get_func_inputs(self, args, kwargs)
        return _unwrap(RunModel(func, func_inputs))

    def train_mode(self, recursive=True):
        self._BaseModel__is_train = True
        for param in self.__arg_params.values():
            param.requires_grad = True

    def infer_mode(self, recursive=True):
        self._BaseModel__is_train = False
        for param in self.__arg_params.values():
            param.requires_grad = False

    def _internal(self, *args, **kwargs):
        """
        Get internal IR information.

        Returns
        -------
        record: _TraceRecord
            The internal record.
            Frontend Model only provides relay function via record.func for now.
        """
        func = self.__train_func if self._BaseModel__is_train else self.__infer_func
        func_inputs = _get_func_inputs(self, args, kwargs, get_handle=False)
        func = annotate_func_params(func, func_inputs)
        requires_grads = [i.requires_grad if isinstance(i, ndarray) else None for i in func_inputs]
        if None in requires_grads:
            requires_grads = []
        named_params = {}
        named_params.update(self.__arg_params)
        named_params.update(self.__aux_params)
        return _TraceRecord(func=func, named_params=named_params, o_struct=None,
                            mutations=None, requires_grads=requires_grads)

    def _state(self):
        state = {}
        state.update(self.__arg_params)
        state.update(self.__aux_params)
        return state

    def to(self, *, device=None, dtype=None):
        for name, param in self.__arg_params.items():
            new_param = param.to(device=device, dtype=dtype)
            self.__arg_params[name] = new_param
        for name, param in self.__aux_params.items():
            new_param = param.to(device=device, dtype=dtype)
            self.__aux_params[name] = new_param

        if self._BaseModel__is_train:
            self.__train_func = AssignDevice(self.__train_func, device)
        else:
            self.__infer_func = AssignDevice(self.__infer_func, device)
