#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Any
from collections import OrderedDict

import mnm
import tvm
import sys
import os

try:
    if sys.argv[1] == '--clean':
        for f in os.listdir('.'):
            if f not in ['main.py', 'util.py'] and f.endswith('.py'):
                os.remove(f'./{f}')
        exit()
except Exception as e:
    pass


@dataclass
class Argument:
    type: str = None
    default: Any = None
    rule: Any = None


def stencil_argument(default=None, rule=None):
    return Argument(type='Union[int, Tuple[int, int]]', default=default, rule=rule)

arith_bin = {
    "args": OrderedDict(
        x1=Argument(type="array_like"),
        x2=Argument(type="array_like"),
    ),
    'ret': 'Union[ndarray, scalar]'
}

config = {
    ('arith', 'op'): {
        "add"          : arith_bin,
        "subtract"     : arith_bin,
        "multiply"     : arith_bin,
        "divide"       : arith_bin,
        "mod"          : arith_bin,
        "less"         : arith_bin,
        "less_equal"   : arith_bin,
        "greater"      : arith_bin,
        "greater_equal": arith_bin,
        "equal"        : arith_bin,
        "not_equal"    : arith_bin,
        "negative": {
            "args": OrderedDict(x=Argument(type="array_like"),),
            'ret': 'Union[ndarray, scalar]'
        },
        'copy': {
            'args': OrderedDict(x=Argument(type='ndarray')),
            'ret': 'ndarray'
        },
    },
    ('nn', 'op'): {
        'avg_pool2d': {
            'args': OrderedDict(input=Argument(type='ndarray'),),
            'attrs': OrderedDict(
                kernel=stencil_argument(),
                stride=stencil_argument(rule='kernel if stride is None else int2tuple(stride)'),
                padding=stencil_argument(0),
                dilation=stencil_argument(1),
                include_pad=Argument(type='bool', default=True),
                ceil_mode=Argument(type='bool', default=False)
            ),
            'ret': 'ndarray'
        },
        'batch_flatten': { 'args': OrderedDict(x=Argument(type='ndarray')), 'ret': 'ndarray'},
        'conv2d': {
            'args': OrderedDict(
                input=Argument(type='ndarray'),
                kernel=Argument(type='ndarray'),
            ),
            'attrs': OrderedDict(
                stride=stencil_argument(1),
                padding=stencil_argument(0),
                dilation=stencil_argument(1),
                groups=Argument(type='int', default=1),
            ),
            'ret': 'ndarray'
        },
        'dropout': {
            'args': OrderedDict(
                x=Argument(type='ndarray'),
            ),
            'attrs': OrderedDict(
                dropout=Argument(type='float'),
                seed=Argument(type='int'),
            ),
            'ret': 'ndarray'
        },
        'max_pool2d': {
            'args': OrderedDict(
                input=Argument(type='ndarray'),
            ),
            'attrs': OrderedDict(
                kernel=stencil_argument(),
                stride=stencil_argument(rule='kernel if stride is None else int2tuple(stride)'),
                padding=stencil_argument(0),
                dilation=stencil_argument(1),
                ceil_mode=Argument(type='bool', default=False)
            ),
            'ret': 'ndarray'
        },
        'batch_norm2d': {
            'args': OrderedDict(
                x=Argument(type='ndarray'),
                mean=Argument(type='ndarray'),
                variance=Argument(type='ndarray'),
                scale=Argument(type='ndarray', default='None',
                    rule='_array([1] * x.shape[1], dtype=x.dtype, ctx=x.ctx) if scale is None else scale'),
                bias=Argument(type='ndarray', default='None',
                    rule='_array([0] * x.shape[1], dtype=x.dtype, ctx=x.ctx) if bias is None else bias'),
            ),
            'attrs': OrderedDict(
                eps=Argument(type='float', default=1e-5),
                momentum=Argument(type='float', default=0.1),
                is_training=Argument(type='float', default=False)
            ),
            'ret': 'ndarray'
        },
        'softmax': {
            'args': OrderedDict(x=Argument(type='ndarray')),
            'attrs': OrderedDict(axis=Argument(type='int')),
            'ret': 'ndarray',
        },
        'log_softmax': {
            'args': OrderedDict(x=Argument(type='ndarray')),
            'attrs': OrderedDict(axis=Argument(type='int')),
            'ret': 'ndarray',
        },
        'relu'   : { 'args': OrderedDict(x=Argument(type='ndarray')), 'ret': 'ndarray' },
        'sigmoid': { 'args': OrderedDict(x=Argument(type='ndarray')), 'ret': 'ndarray' },
        'tanh'   : { 'args': OrderedDict(x=Argument(type='ndarray')), 'ret': 'ndarray' },
    }
}

undumped = set(mnm._core.op.OP_DICT.keys())
generated = set()

import util

attrs_name = {}

f_get_op = mnm._ffi._tvm._get_global_func('relay.op._GetOp')
for name in map(lambda x: x.value, mnm._ffi._tvm._get_global_func('relay.op._ListOpNames')()):
    attrs_name[name] = f_get_op(name).attrs_type_key


for k, ops in config.items():
    module, prfx = k
    with open(f'{module}.py', 'w') as f_module:
        generated.add(f_module.name)
        f_module.write('from typing import Union, Tuple\n')
        f_module.write('from .._core.ndarray import ndarray\n')
        f_module.write('from .._core.op import get_op\n')
        f_module.write('from .._ffi._tvm import _make_node\n')
        f_module.write('from ._typing import array_like, scalar\n')
        f_module.write('from ._util import int2tuple\n')
        f_module.write('from .imports import array as _array\n')
        f_module.write('from ._typing import _ARG_TYPE_GUARDS, _RET_TYPE_GUARDS\n\n')
        f_module.write('\n')
        for op, rule in ops.items():
            op_key = f'mnm.{prfx}.{op}'
            undumped.remove(op_key)
            args = rule['args']
            attrs = rule.get('attrs', OrderedDict())
            arg_list = args.copy()
            arg_list.update(attrs)
            arg_str = (',\n' + ' ' * (5 + len(op))).join(util.to_arg_list(arg_list))
            f_module.write(f'def {op}({arg_str}) -> {rule["ret"]}:\n')
            f_module.write(f'    """ Reserved for doc string... """\n')
            for name, arg in arg_list.items():
                if arg.rule is not None:
                    f_module.write(f'    {name} = {arg.rule}\n')
                elif arg.type == 'Union[int, Tuple[int, int]]':
                    f_module.write(f'    {name} = int2tuple({name})\n')

            op_entry = mnm._core.op.OP_DICT[op_key]


            attr_str = 'None'
            if attrs_name[op_key]:
                f_module.write('    attr_args = {\n')
                for arg in rule.get('attrs', OrderedDict()).keys():
                    f_module.write(f'        "{arg}": {arg},\n')
                f_module.write('    }\n')
                attr_str = f'_make_node("{attrs_name[op_key]}", **attr_args)'

            param_str = ', '.join(i for i in rule['args'].keys())

            for name, arg in rule['args'].items():
                f_module.write(f'    {name} = _ARG_TYPE_GUARDS[{arg.type}]({name}, "{name}")\n')
            f_module.write(f'    f = get_op("{op_key}")\n')
            f_module.write(f'    res = f(eager=True,\n            args=[{param_str}],\n            attrs={attr_str})\n')
            f_module.write(f'    res = _RET_TYPE_GUARDS[{rule["ret"]}](res, "return value")\n')
            f_module.write(f'    return res\n\n')

print('Unused: ', undumped)


try:
    move = sys.argv[1] != '--local'
except:
    move = True
    pass

if move:
    import shutil
    for i in generated:
        shutil.move(i, f'../../python/mnm/op/{i}')
