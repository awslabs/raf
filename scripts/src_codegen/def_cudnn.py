#!/usr/bin/env python3

from collections import OrderedDict
import os
import itertools

from . import codegen_utils

def strip_type_commons(s):
    return s[len('cudnn'):-len('Descriptor_t')]

def call_cudnn_api(api, args):
    return f'CUDNN_CALL({"" if api.startswith("cudnn") else "cudnn"}{api}({args}));'

class Status(object):

    def __init__(self, schema, cudnn_apis, wrappers, symbols):
        self.schema = schema
        self.cudnn_apis = cudnn_apis
        self.symbols = symbols
        self.attrs = dict()
        self.casts = dict()
        self.wrappers = wrappers

    def add_attr(self, ty, name):
        assert name not in self.attrs.keys(), name
        self.attrs[name] = ty

    def add_wrapper(self, key, func):
        assert key not in self.wrappers.keys()
        self.wrappers[key] = func

    def add_cast(self, ty, name, value):
        self.casts[name] = (ty, value)

    def set_hardcode(self, k, v):
        if k in self.symbols.keys():
            msg = f"You cannot override existing hardcode! {k}, {self.symbols[k]}, {v}"
            assert False, msg
        self.symbols[k] = v


    def get_destructors(self):
        res = [call_cudnn_api(f'Destroy{strip_type_commons(ty)}Descriptor', name)
               for name, ty in self.attrs.items() if ty.endswith('Descriptor_t')]
        return '\n    '.join(res)

    def get_attrs(self):
        return '\n  '.join(f'{ty} {name};' for name, ty in self.attrs.items())

    def get_casts(self):
        casts = [f'{v[0]} {name} = {v[1]};\n    (void) {name};' for name, v in self.casts.items()]
        return '\n    '.join(casts)


class ShapeWrapper(object):

    def __init__(self, name, tensor, flatten=None, need_stride=True):
        self.name = name
        self.tensor = tensor
        self.flatten = flatten
        self.need_stride = need_stride

    def add_cast(self, status):
        assert '->' in self.tensor, self.tensor
        name = self.tensor.split('->')[1]
        status.add_cast('DLTensor*', name, self.tensor)
        self.tensor = name

    def normalize(self, status):
        self.add_cast(status)
        slices = ''
        if self.flatten is not None:
            slices = ', '.join(str(i).replace('<last>', f'{self.tensor}->ndim') for i in self.flatten)
        res = [f'auto {self.name}_tt = SquashTensorShape({self.tensor}, {{{slices}}});']
        return res

class CUDNNFilter(ShapeWrapper):

    # `flatten' is only for legacy normalization.
    def __init__(self, name, ptr, tensor, flatten=None):
        assert flatten is None
        ShapeWrapper.__init__(self, name, tensor, flatten, False)
        self.ptr = ptr

    def normalize(self, status):
        ShapeWrapper.add_cast(self, status)
        status.add_attr(f'cudnnFilterDescriptor_t', self.name)
        status.set_hardcode(self.ptr, f'{self.tensor}->data')
        res = [f'auto {self.name}_tt = SquashTensorShape({self.tensor}, {{}});']
        res += [f'(void) {self.name}_tt;']
        res += [f'{self.name} = NormalizeFilter({self.tensor});']
        return res


class CUDNNTensor(ShapeWrapper):

    def __init__(self, desc, ptr, tensor, flatten=None):
        ShapeWrapper.__init__(self, desc, tensor, flatten, True)
        self.ptr = ptr

    def normalize(self, status):
        res = ShapeWrapper.normalize(self, status)
        status.set_hardcode(self.ptr, f'{self.tensor}->data')
        status.add_attr(f'cudnnTensorDescriptor_t', self.name)
        res += [f'{self.name} = NormalizeTensorType({self.name}_tt);']
        return res

def CUDNNOutput(Base):

    class Output(Base):

        def __init__(self, desc, ptr, tensor, flatten=None):
            Base.__init__(self, desc, ptr, tensor, flatten)
            self.ptr = ptr

        def add_cast(self, status):
            if self.tensor == 'cv->out':
                status.add_cast('DLTensor*', 'out', 'cv->out')
                self.tensor = 'out'
            elif self.tensor.startswith('tv->fields['):
                idx = int(self.tensor[len('tv->fields['):].rstrip(']'))
                status.add_cast('DLTensor*', f'out{idx}', self.tensor)
                self.tensor = f'out{idx}'
            else:
                assert False, self.tensor

        def normalize(self, status):
            res = Base.normalize(self, status)
            return res

    return Output

CUDNNOutputTensor = CUDNNOutput(CUDNNTensor)
CUDNNOutputFilter = CUDNNOutput(CUDNNFilter)

class CUDNNDescriptor(object):

    def __init__(self, name, args, ty, setter=None):
        self.name = name
        self.args = args
        self.ty = ty
        self.setter = f'cudnnSet{ty}Descriptor' if setter is None else setter

    def normalize(self, status):
        creator = f'cudnnCreate{self.ty}Descriptor'
        assert creator in status.cudnn_apis.keys()
        assert self.setter in status.cudnn_apis.keys()
        status.add_attr(f'cudnn{self.ty}Descriptor_t', self.name)
        # Here I assume target is always the first argument
        args = [self.name] + list(self.args)
        res = [call_cudnn_api(creator, f'&{self.name}'),
               call_cudnn_api(self.setter, ', '.join(args))]
        return res

class CUDNNAlgorithm(object):

    def __init__(self, perf_ty, name, api, args, keys):
        self.perf_ty = perf_ty
        self.name = name
        self.api = api
        self.args = args
        self.keys = keys

    def normalize(self, status):
        fmt = """
MetaCache<{PERF_T}> {CACHE};
{PERF_T} Find{PERF_T}Wrapper(const std::vector<uint8_t> &key, {ARGS}) {{
  std::lock_guard<std::mutex> lock({CACHE}.mu);
  if (auto *val = {CACHE}.Get(key)) {{
    return *val;
  }}
  int cnt;
  {PERF_T} res;
  {FINDER}
  if (res.status != CUDNN_STATUS_SUCCESS) {{
    LOG(FATAL) << "ValueError: Cannot find a proper algorithm " << cudnnGetErrorString(res.status);
    throw;
  }}
  {CACHE}.Set(key, res);
  return res;
}}
""".strip()

        api_args = status.cudnn_apis[f'cudnn{self.api}']
        decl_args = [str(arg) for arg in api_args[1:] if arg.name not in self.args.keys()]
        decl_args = ', '.join(decl_args)
        site_params = [arg.name for arg in api_args[1:] if arg.name not in self.args.keys()]
        site_params = ', '.join(site_params)

        params = ['CUDNNThreadEntry::ThreadLocal()->handle']
        for arg in api_args[1:]:
            name = arg.name
            if name in status.symbols.keys():
                params.append(status.symbols[name])
            elif name in status.attrs.keys():
                params.append(name)
            elif name in self.args.keys():
                params.append(str(self.args[name]))
            else:
                assert False, f'{name} is missing!'
        finder_params = ', '.join(params)

        if self.perf_ty not in status.wrappers.keys():
            status.add_wrapper(
                    self.perf_ty,
                    fmt.format(PERF_T=self.perf_ty,
                               ARGS=decl_args,
                               FINDER=call_cudnn_api(self.api, finder_params),
                               CACHE=f'CacheFor{self.perf_ty}'))

        res = [f'HashKey {self.name}_hasher;']
        res += [f'{self.name}_hasher << ' + ' << '.join(self.keys) + ';']
        res += [f'const auto &{self.name}_key = {self.name}_hasher.byte_vector;']
        res += [f'{self.name} = Find{self.perf_ty}Wrapper({self.name}_key, {site_params});']
        status.add_attr(self.perf_ty, self.name)
        return res


class CUDNNWorkSpace(object):

    def __init__(self, size, ptr, api, args):
        self.size = size
        self.ptr = ptr
        self.api = api
        self.args = ['CUDNNThreadEntry::ThreadLocal()->handle'] + args

    def normalize(self, status):
        status.add_attr('size_t', self.size)
        status.add_attr('void *', self.ptr)
        res = [call_cudnn_api(self.api, ', '.join(self.args))]
        res += [f'RequestWorkspace(&{self.ptr}, cv->ctx, {self.size});']
        return res

class AssignStatement(object):

    def __init__(self, ty, name, value, is_attr):
        self.ty = ty
        self.name = name
        self.value = value
        self.is_attr = is_attr

    def normalize(self, status):
        if self.is_attr:
            status.add_attr(self.ty, self.name)
            return [f'{self.name} = {self.value};']
        if not self.ty and not self.name:
            return [self.value + ';']
        return [f'{self.ty} {self.name} = {self.value};']

class CUDNNDispatch(object):
    fmt = """
class {CLASSNAME} : public mnm::op::OpEnv {{
  {ATTRS}
  explicit {CLASSNAME}(const CallValues &cv) {{
    {CASTS}
    {CONSTRUCTOR}
  }}
 public:
  ~{CLASSNAME}() {{
    {DESTRUCTOR}
  }}
  void Execute(const CallValues &cv) {{
    {CASTS}
    {INVOKE}
  }}
  static OpEnv *make(const CallValues &cv) {{
    return new {CLASSNAME}(cv);
  }}
}};

MNM_OP_DISPATCH("mnm.op.{OP}", {CLASSNAME}::make, DevType::kCUDA(), "generated_cudnn");
""".strip()


    def __init__(self, op, api, arg_type, normalizers, args, output=['DLTensor *']):
        self.op = op
        self.api = api
        self.arg_type = arg_type
        self.normalizers = normalizers
        self.args = args.copy()
        self.output = output

    def normalize(self, ops, schema, cudnn_apis, wrappers):
        class_name = f'{codegen_utils.snake_to_pascal(self.op)}ImplementedByCUDNN{self.api}'
        status = Status(schema[ops[self.op].schema_name], cudnn_apis, wrappers, self.args)
        constructor = [i.normalize(status) for i in self.normalizers]
        constructor = itertools.chain.from_iterable(constructor)
        constructor = '\n    '.join(constructor)
        invoke = ['CUDNNThreadEntry::ThreadLocal()->handle']
        for elem in cudnn_apis[f'cudnn{self.api}'][1:]:
            name = elem.name
            if name in self.args.keys():
                invoke.append(self.args[name])
            elif name in status.attrs.keys():
                if name == "algo":
                    invoke.append("algo.algo")
                else:
                    invoke.append(str(name))
            else:
                assert False, f'{name} is missing!'
        output = ''
        if len(self.output) != 1:
            output = 'auto *tv = const_cast<TupleValueObj*>(cv->out.as<TupleValueObj>());\n'
        casts = """
    auto args = cv->args.as<mnm::op::schema::{TYPE}>();
    (void) args;
    {CASTS}
""".strip().format(TYPE=codegen_utils.snake_to_pascal(self.arg_type)+"Args",
                   CASTS=output + status.get_casts())
        invoke = call_cudnn_api(self.api, ', '.join(invoke))
        return self.fmt.format(OP=self.op,
                               CLASSNAME=class_name,
                               ATTRS=status.get_attrs(),
                               CONSTRUCTOR=constructor,
                               DESTRUCTOR=status.get_destructors(),
                               CASTS=casts,
                               INVOKE=invoke)


def constant(v, ty_src='out->dtype'):
    return f'CUDNNDType({ty_src}).const_addr<{v}>()'

ALPHA_BETA = {'alpha': constant(1), 'beta': constant(0)}

def dispatch_unary(op, enum):
    x = CUDNNTensor('xDesc', 'x', 'args->x', [0, '<last>'])
    y = CUDNNOutputTensor('yDesc', 'y', 'cv->out', [0, '<last>'])
    act_desc = CUDNNDescriptor('activationDesc', [enum, 'CUDNN_PROPAGATE_NAN', '0.0'], 'Activation')
    return CUDNNDispatch(op, 'ActivationForward', 'unary', [x, y, act_desc], ALPHA_BETA)

def dispatch_unary_dx(op, enum):
    x = CUDNNTensor('xDesc', 'x', 'args->x', [0, '<last>'])
    y = CUDNNTensor('yDesc', 'y', 'args->y', [0, '<last>'])
    dy = CUDNNTensor('dyDesc', 'dy', 'args->dy', [0, '<last>'])
    dx = CUDNNOutputTensor('dxDesc', 'dx', 'cv->out', [0, '<last>'])
    act_desc = CUDNNDescriptor('activationDesc', [enum, 'CUDNN_PROPAGATE_NAN', '0.0'], 'Activation')
    return CUDNNDispatch(op, 'ActivationBackward', 'unary_dx', [x, y, dy, dx, act_desc], ALPHA_BETA)

def dispatch_softmax(op, algorithm):
    consts = ALPHA_BETA.copy()
    consts['algo'] = algorithm
    axis = AssignStatement('int', 'axis', '(args->axis + x->ndim) % x->ndim', False)
    x = CUDNNTensor('xDesc', 'x', 'args->x', [0, 'axis', 'axis+1', '<last>'])
    y = CUDNNOutputTensor('yDesc', 'y', 'cv->out', [0, 'axis', 'axis+1', '<last>'])
    mode = """GetTensorTypeDim(xDesc_tt, 1) == 1 && GetTensorTypeDim(xDesc_tt, 2) == 1 ?
    CUDNN_SOFTMAX_MODE_INSTANCE : CUDNN_SOFTMAX_MODE_CHANNEL"""
    mode = AssignStatement('cudnnSoftmaxMode_t', 'mode', mode, True)
    return CUDNNDispatch(op, 'SoftmaxForward', 'softmax', [axis, x, y, mode], consts)

def dispatch_softmax_dx(op, algorithm):
    consts = ALPHA_BETA.copy()
    consts['algo'] = algorithm
    axis = AssignStatement('int', 'axis', '(args->axis + x->ndim) % x->ndim', False)
    x = CUDNNTensor('xDesc', 'x', 'args->x', [0, 'axis', 'axis+1', '<last>'])
    y = CUDNNTensor('yDesc', 'y', 'args->y', [0, 'axis', 'axis+1', '<last>'])
    dy = CUDNNTensor('dyDesc', 'dy', 'args->dy', [0, 'axis', 'axis+1', '<last>'])
    dx = CUDNNOutputTensor('dxDesc', 'dx', 'cv->out', [0, 'axis', 'axis+1', '<last>'])
    mode = """GetTensorTypeDim(xDesc_tt, 1) == 1 && GetTensorTypeDim(xDesc_tt, 2) == 1 ?
    CUDNN_SOFTMAX_MODE_INSTANCE : CUDNN_SOFTMAX_MODE_CHANNEL"""
    mode = AssignStatement('cudnnSoftmaxMode_t', 'mode', mode, True)
    return CUDNNDispatch(op, 'SoftmaxBackward', 'softmax_dx', [axis, x, y, dy, dx, mode], consts)

def normalize_scalar_to_tuple(dim, name):
    return f'CastVector<int, int64_t>(NormalizeScalarToTuple<{dim}>({name}))'

def dispatch_pooling(op, dims, mode):
    x = CUDNNTensor('xDesc', 'x', 'args->x')
    y = CUDNNOutputTensor('yDesc', 'y', 'cv->out')
    stencil_attrs = ['kernel', 'stride', 'padding', 'dilation']
    stencil_args = [AssignStatement('std::vector<int>', attr,
                                    normalize_scalar_to_tuple(dims, f'args->{attr}'), False)
                                    for attr in stencil_attrs]
    desc_args = [mode, 'CUDNN_PROPAGATE_NAN', '2', 'BeginPtr(kernel)',
                 'BeginPtr(padding)', 'BeginPtr(stride)']
    pool_desc = CUDNNDescriptor('poolingDesc', desc_args, 'Pooling',
                                setter='cudnnSetPoolingNdDescriptor')
    args = [x, y] + stencil_args + [pool_desc]
    return CUDNNDispatch(op, 'PoolingForward', 'pool', args, ALPHA_BETA)

def dispatch_pooling_dx(op, dims, mode):
    x = CUDNNTensor('xDesc', 'x', 'args->x')
    y = CUDNNTensor('yDesc', 'y', 'args->y')
    dy = CUDNNTensor('dyDesc', 'dy', 'args->dy')
    dx = CUDNNOutputTensor('dxDesc', 'dx', 'cv->out')
    stencil_attrs = ['kernel', 'stride', 'padding', 'dilation']
    stencil_args = [AssignStatement('std::vector<int>', attr,
                                    normalize_scalar_to_tuple(dims, f'args->{attr}'), False)
                                    for attr in stencil_attrs]
    desc_args = [mode, 'CUDNN_PROPAGATE_NAN', '2', 'BeginPtr(kernel)',
                 'BeginPtr(padding)', 'BeginPtr(stride)']
    pool_desc = CUDNNDescriptor('poolingDesc', desc_args, 'Pooling',
                                setter='cudnnSetPoolingNdDescriptor')
    args = [x, y, dy, dx] + stencil_args + [pool_desc]
    return CUDNNDispatch(op, 'PoolingBackward', 'pool_dx', args, ALPHA_BETA)

def dispatch_batchnorm_dxwb(op):
    x = CUDNNTensor('xDesc', 'x', 'args->x', [0, 1, 2, 3, '<last>'])
    dy = CUDNNTensor('dyDesc', 'dy', 'args->dy', [0, 1, 2, 3, '<last>'])
    dx = CUDNNOutputTensor('dxDesc', 'dx', 'tv->fields[0]', [0, 1, 2, 3, '<last>'])
    dw = CUDNNOutputTensor('dBnScaleBiasDesc', 'dBnScaleResult', 'tv->fields[1]', [0, 0, 1, '<last>'])

    args = {
      'alphaDataDiff': constant(1, 'out0->dtype'),
      'alphaParamDiff': constant(1, 'out1->dtype'),
      'betaDataDiff': constant(0, 'out0->dtype'),
      'betaParamDiff': constant(0, 'out1->dtype'),

      'mode': 'CUDNN_BATCHNORM_SPATIAL',
      'epsilon': 'args->eps',
      'savedMean': 'nullptr',
      'savedInvVariance': 'nullptr',
      'bnScale': '(args->w).as<value::TensorValueObj>()->tensor->data',
      'dBnBiasResult': 'tv->fields[2].operator DLTensor*()->data',
    }

    return CUDNNDispatch(op, 'BatchNormalizationBackward', 'batch_norm_train_dxwb', [x, dy, dx, dw], args, ['DLTensor *'] * 3)

def dispatch_batchnorm(op, api):
    x = CUDNNTensor('xDesc', 'x', 'args->x', [0, 1, 2, 3, '<last>'])
    w = CUDNNTensor('bnScaleBiasMeanVarDesc', 'bnScale', 'args->w', [0, 0, 1, '<last>'])
    if op.endswith('_train'):
        y = CUDNNOutputTensor('yDesc', 'y', 'tv->fields[0]', [0, 1, 2, 3, '<last>'])
        out_types = ['DLTensor *'] * 3
    else:
        y = CUDNNOutputTensor('yDesc', 'y', 'cv->out', [0, 1, 2, 3, '<last>'])
        out_types = ['DLTensor *']
    args = {
      'alpha': constant(1, 'x->dtype'),
      'beta': constant(0, 'x->dtype'),
      'mode': 'CUDNN_BATCHNORM_SPATIAL',
      'bnBias': '(args->b).as<value::TensorValueObj>()->tensor->data',
      'epsilon': 'args->eps',
      'estimatedMean': '(args->running_mean).as<value::TensorValueObj>()->tensor->data',
      'estimatedVariance': '(args->running_var).as<value::TensorValueObj>()->tensor->data',
      'resultRunningMean': '(args->running_mean).as<value::TensorValueObj>()->tensor->data',
      'resultRunningVariance': '(args->running_var).as<value::TensorValueObj>()->tensor->data',
      'exponentialAverageFactor': 'args->momentum',
      'resultSaveMean': 'nullptr',
      'resultSaveVariance': 'nullptr',
      'resultSaveInvVariance': 'nullptr',
    }

    return CUDNNDispatch(op, api, 'batch_norm', [x, y, w], args, out_types)

def dispatch_conv(op, dims):
    x = CUDNNTensor('xDesc', 'x', 'args->x')
    w = CUDNNFilter('wDesc', 'w', 'args->w')
    y = CUDNNOutputTensor('yDesc', 'y', 'cv->out')
    stencil_attrs = ['stride', 'padding', 'dilation']
    stencil_args = [AssignStatement('std::vector<int>', attr,
                                    normalize_scalar_to_tuple(dims, f'args->{attr}'), False)
                                    for attr in stencil_attrs]
    conv_desc = CUDNNDescriptor('convDesc', [f'{dims}', 'BeginPtr(padding)', 'BeginPtr(stride)',
        'BeginPtr(dilation)', 'CUDNN_CROSS_CORRELATION', 'CUDNNDType(w->dtype)'],
        'Convolution', setter='cudnnSetConvolutionNdDescriptor')
    pre_mathtype = AssignStatement('', '', 'if (ir::DataType(w->dtype).is_float16()) {cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);}', False)
    group = AssignStatement('', '', 'cudnnSetConvolutionGroupCount(convDesc, args->groups)', False)
    algo = CUDNNAlgorithm(name='algo',
                          perf_ty='cudnnConvolutionFwdAlgoPerf_t',
                          api='FindConvolutionForwardAlgorithm',
                          args={
                               'requestedAlgoCount': 1,
                               'returnedAlgoCount': '&cnt',
                               'perfResults': '&res'
                          },
                          keys=[
                            'args->stride',
                            'args->padding',
                            'args->dilation',
                            'wDesc_tt',
                            'xDesc_tt',
                            'yDesc_tt',
                          ])
    ws = CUDNNWorkSpace('workSpaceSizeInBytes', 'workSpace', 'GetConvolutionForwardWorkspaceSize',
            ['xDesc', 'wDesc', 'convDesc', 'yDesc', 'algo.algo', '&workSpaceSizeInBytes'])
    post_mathtype = AssignStatement('', '', 'cudnnSetConvolutionMathType(convDesc, algo.mathType)', False)
    normalizers = [x, w, y, ] + stencil_args + [conv_desc, pre_mathtype, group, algo, ws, post_mathtype]
    return CUDNNDispatch(op=op,
                         api='ConvolutionForward',
                         arg_type='conv',
                         normalizers=normalizers,
                         args={
                             'alpha': constant(1),
                             'beta': constant(0),
                         })

def dispatch_conv_dxw(op, xorw, dims):
    if xorw == 'Filter':
        x = CUDNNTensor('xDesc', 'x', 'args->x_or_w')
        w = CUDNNOutputFilter(f'dwDesc', f'dw', 'cv->out')
        x_or_w, diff = x, w
    else:
        x = CUDNNOutputTensor(f'dxDesc', f'dx', 'cv->out')
        w = CUDNNFilter('wDesc', 'w', 'args->x_or_w')
        x_or_w, diff = w, x

    dy = CUDNNTensor('dyDesc', 'dy', 'args->dy')

    stencil_attrs = ['stride', 'padding', 'dilation']
    stencil_args = [AssignStatement('std::vector<int>', attr,
                                    normalize_scalar_to_tuple(dims, f'args->{attr}'), False)
                                    for attr in stencil_attrs]
    conv_desc = CUDNNDescriptor('convDesc', [f'{dims}', 'BeginPtr(padding)', 'BeginPtr(stride)',
        'BeginPtr(dilation)', 'CUDNN_CROSS_CORRELATION', 'CUDNNDType(x_or_w->dtype)'], 'Convolution',
        setter='cudnnSetConvolutionNdDescriptor')
    pre_mathtype = AssignStatement('', '', 'if (ir::DataType(x_or_w->dtype).is_float16()) {cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);}', False)
    group = AssignStatement('', '', 'cudnnSetConvolutionGroupCount(convDesc, args->groups)', False)
    algo = CUDNNAlgorithm(name='algo',
                          perf_ty=f'cudnnConvolutionBwd{xorw}AlgoPerf_t',
                          api=f'FindConvolutionBackward{xorw}Algorithm',
                          args={
                               'requestedAlgoCount': 1,
                               'returnedAlgoCount': '&cnt',
                               'perfResults': '&res'
                          },
                          keys=[
                            'args->stride',
                            'args->padding',
                            'args->dilation',
                            f'{x_or_w.name}_tt',
                            'dyDesc_tt',
                            f'{diff.name}_tt',
                          ])
    ws = CUDNNWorkSpace('workSpaceSizeInBytes', 'workSpace', f'GetConvolutionBackward{xorw}WorkspaceSize',
            [f'{x_or_w.name}', 'dyDesc', 'convDesc', f'{diff.name}', 'algo.algo', '&workSpaceSizeInBytes'])
    post_mathtype = AssignStatement('', '', 'cudnnSetConvolutionMathType(convDesc, algo.mathType)', False)
    normalizers = [x, w, dy] + stencil_args + [conv_desc, pre_mathtype, group, algo, ws, post_mathtype]
    return CUDNNDispatch(op=op,
                         api=f'ConvolutionBackward{xorw}',
                         arg_type='conv_dxw',
                         normalizers=normalizers,
                         args=ALPHA_BETA)


SCHEMAS = [
    dispatch_unary('relu', 'CUDNN_ACTIVATION_RELU'),
    dispatch_unary('tanh', 'CUDNN_ACTIVATION_TANH'),
    dispatch_unary('sigmoid', 'CUDNN_ACTIVATION_SIGMOID'),
    dispatch_softmax('softmax', 'CUDNN_SOFTMAX_ACCURATE'),
    dispatch_softmax('log_softmax', 'CUDNN_SOFTMAX_LOG'),
    dispatch_pooling('max_pool2d', 2, 'CUDNN_POOLING_MAX'),
    dispatch_pooling('avg_pool2d', 2, """args->include_pad ?
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING"""),
    dispatch_batchnorm('batch_norm_infer', 'BatchNormalizationForwardInference'),
    dispatch_batchnorm('batch_norm_train', 'BatchNormalizationForwardTraining'),
    dispatch_conv('conv2d', 2),
    # Backwards
    dispatch_unary_dx('relu_dx', 'CUDNN_ACTIVATION_RELU'),
    dispatch_unary_dx('tanh_dx', 'CUDNN_ACTIVATION_TANH'),
    dispatch_unary_dx('sigmoid_dx', 'CUDNN_ACTIVATION_SIGMOID'),
    dispatch_softmax_dx('softmax_dx', 'CUDNN_SOFTMAX_ACCURATE'),
    dispatch_softmax_dx('log_softmax_dx', 'CUDNN_SOFTMAX_LOG'),
    dispatch_pooling_dx('max_pool2d_dx', 2, 'CUDNN_POOLING_MAX'),
    dispatch_pooling_dx('avg_pool2d_dx', 2, """args->include_pad ?
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING"""),
    dispatch_conv_dxw('conv2d_dx', 'Data', 2),
    dispatch_conv_dxw('conv2d_dw', 'Filter', 2),
    dispatch_batchnorm_dxwb('batch_norm_train_dxwb'),

]


#('dropout', {
#    'dlts'                    : """
#    int n = args.size();
#    std::vector<const DLTensor*> dlts(n);
#    for (int i = 0; i < n - 1; ++i) {
#      dlts[i] = args[i];
#    }
#    value::TupleValue tv = Downcast<TupleValue>(args[n - 1]);
#    dlts[n - 1] = tv->fields[0];
#    std::vector<OpaqueValue> opqs{Downcast<OpaqueValue>(tv->fields[1]),
#                                  Downcast<OpaqueValue>(tv->fields[2])};
#""",
#    'callee'       : 'cudnnDropoutForward',
#    'attrs_t'      : 'DropoutAttrs',
#    'dropoutDesc'  : {'callee'  : 'cudnnSetDropoutDescriptor',
#                      'dropout' : 'casted_ptr->dropout',
#                      'seed'    : 'casted_ptr->seed',
#                      'states'  : {'reserve' : 'cudnnDropoutGetStatesSize',
#                                   'res'     : 'opqs[0]',
#                                   'size_t'  : 'stateSizeInBytes'},},
#    'reserveSpace' : {'reserve' : 'cudnnDropoutGetReserveSpaceSize',
#                      'res'     : 'opqs[1]'},
#}),
#('grad.dropout', {
#    'dlts'                    : """
#    int n = args.size();
#    std::vector<const DLTensor*> dlts(n);
#    for (int i = 0; i < n - 1; ++i) {
#      dlts[i] = args[i];
#    }
#    value::TupleValue tv = Downcast<TupleValue>(args[n - 1]);
#    dlts[n - 1] = tv->fields[0];
#    std::vector<OpaqueValue> opqs{Downcast<OpaqueValue>(tv->fields[1]),
#                                  Downcast<OpaqueValue>(tv->fields[2])};
#""",
#    'callee'                  : 'cudnnDropoutBackward',
#    'attrs_t'                 : 'DropoutAttrs',
#    'dropoutDesc'             : {'callee'           : 'cudnnRestoreDropoutDescriptor',
#                                 'dropout'          : 'casted_ptr->dropout',
#                                 'seed'             : 'casted_ptr->seed',
#                                 'states'           : 'Downcast<BufferValue>(opqs[0])->data',
#                                 'stateSizeInBytes' : 'Downcast<BufferValue>(opqs[0])->size_in_bytes'},
#    'reserveSpace'            : 'Downcast<BufferValue>(opqs[1])->data',
#    'reserveSpaceSizeInBytes' : 'Downcast<BufferValue>(opqs[1])->size_in_bytes',
#}),
