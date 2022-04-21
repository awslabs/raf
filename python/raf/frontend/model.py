# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model that wraps the IR converted from deep learning frameworks."""
# pylint: disable=protected-access, attribute-defined-outside-init
from tvm import relay

from raf._ffi.model import RunModel
from raf.model.model import BaseModel
from raf.model.trace import _unwrap, _TraceRecord
from raf._core.ir_ext import extended_var
from raf._core.ndarray import ndarray, Symbol
from raf._ffi.pass_ import AssignDevice, ExprAppend, ExtractBinding, InferType, Substitute
from raf._core.module import IRModule


def _get_main_func_params(model, args, kwargs, get_handle=True):
    arg_index = 0
    res = []
    mod = (
        model._FrameworkModel__train_mod
        if model._BaseModel__is_train
        else model._FrameworkModel__infer_mod
    )
    func = mod["main"]
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


def _get_func_output_var(func):
    body = func.body
    while not isinstance(body, relay.Var):
        if isinstance(body, relay.Let):
            body = body.body
        else:
            raise NotImplementedError("Not supported type: ", type(body))
    return body


def annotate_main_func_params(func, args):
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

    params = [extended_var(param.name_hint, get_type(arg)) for param, arg in zip(func.params, args)]
    vmap = dict(zip(func.params, params))
    body = Substitute(func.body, vmap)
    return relay.Function(params, body)


class FrameworkModel(BaseModel):
    """Represent the wrapper of the models read from deep learning frameworks.

    Parameters
    ----------
    train_mod : RAF module
        Module contains both forward and backward computation.

    infer_mod : RAF module
        Forward module

    arg_params : Dict[str, ndarray]
        Model parameters

    aux_params : Dict[str, ndarray]
        Auxiliary params, not learnable
    """

    # pylint: disable=invalid-name

    def __init__(self, train_mod, infer_mod, arg_params, aux_params):
        super(FrameworkModel, self).__init__()
        assert isinstance(train_mod, IRModule)
        assert isinstance(infer_mod, IRModule)
        self.__train_mod = train_mod
        self.__infer_mod = infer_mod
        self.__arg_params = arg_params
        self.__aux_params = aux_params
        for param in self.__aux_params.values():
            param.requires_grad = False
        self.__recorded = None

    def __call__(self, *args, **kwargs):
        mod = self.__infer_mod
        if self._BaseModel__is_train:
            mod = self.__train_mod
        func_inputs = _get_main_func_params(self, args, kwargs)
        return _unwrap(RunModel(mod, func_inputs))

    def record(self, *args, **kwargs):
        """
        Get the return symbol of the function

        Returns
        -------
        ret: Symbol
            The return symbol, via which user can build new layers and append to existing Model.
        """
        r = self._internal(*args, **kwargs)
        # (function, num_argument)
        mod = InferType()(r.mod)
        self.__recorded = (mod, len(args) + len(kwargs))
        ret_var = _get_func_output_var(mod["main"])
        ret = Symbol.from_expr(ret_var)
        if isinstance(ret_var.checked_type, relay.TupleType):
            ret = ret[0]
        return ret

    def __add__(self, other):
        if not self.__recorded:
            raise ValueError("Call model.record first to get the output symbols.")
        mod, num_orig_arg = self.__recorded
        func = mod["main"]
        ret_var = _get_func_output_var(func)
        if isinstance(ret_var.checked_type, relay.TupleType):
            ret = Symbol.from_expr(ret_var)
            other = [
                other,
            ] + [ret[i + 1] for i in range(len(ret_var.checked_type.fields) - 1)]
            other = Symbol.make_tuple(other)
        other = ExtractBinding(other._Symbol__handle, [ret_var])
        new_body = ExprAppend(func.body, other)
        free_vars = relay.analysis.free_vars(new_body)
        input_params = func.params[:num_orig_arg]
        new_free_vars = input_params[:]
        # Reorder the params so as to guarantee inputs to be at the begining of array
        for var in free_vars:
            if var not in input_params:
                new_free_vars.append(var)
        # [arguments, parameters]
        new_params = (
            new_free_vars[0:num_orig_arg]
            + new_free_vars[len(func.params) :]
            + func.params[num_orig_arg:]
        )
        new_func = relay.Function(new_params, new_body)
        new_mod = IRModule.from_expr(new_func)
        return FrameworkModel(new_mod, new_mod, self.__arg_params, self.__aux_params)

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
            Frontend Model only provides relay function via record.mod for now.
        """
        mod = self.__train_mod if self._BaseModel__is_train else self.__infer_mod
        func_inputs = _get_main_func_params(self, args, kwargs, get_handle=False)
        mod["main"] = annotate_main_func_params(mod["main"], func_inputs)
        requires_grads = [i.requires_grad if isinstance(i, ndarray) else None for i in func_inputs]
        if None in requires_grads:
            requires_grads = []
        named_params = {}
        named_params.update(self.__arg_params)
        named_params.update(self.__aux_params)
        return _TraceRecord(
            mod=mod,
            named_params=named_params,
            o_struct=None,
            mutations=None,
            requires_grads=requires_grads,
        )

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

        self.__train_mod = AssignDevice(device)(self.__train_mod)
        self.__infer_mod = AssignDevice(device)(self.__infer_mod)
