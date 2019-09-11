#!/usr/bin/env python3

import pycparser
import os
import sys
import shutil
import subprocess

from collections import OrderedDict

if len(sys.argv) != 2:
    print("Usage: python cudnn_op_gen.py <generated-file-name>")
    print("Note: ensure CUDA_HOME and CUDNN_HOME are in your environment")
    quit()

file_name = sys.argv[1]

cudnn_home = os.getenv("CUDNN_HOME")
cuda_home = os.getenv("CUDA_HOME")
cudnn_h = "%s/include/cudnn.h" % cudnn_home

import preprocessor
src = preprocessor.cleanup(cudnn_h, "-I%s/include" % cuda_home)

import extractor
funcs = extractor.extract_functions(src)

TO_INT_PTR = 'BeginPtr(MakeShape<int>(%s))'

"""
Rules of generating cuDNN backend:
    1. Each record of rule is a (operator name, root rule) tuple. Operator name is a `str`, and
       root rule is a `dict` (refer 2. for more details).
    2. The "root rule" indicates the rule of preparing parameters for calling the given
       cuDNN API, including:
       2.1. 'callee': Indicates the function to be called to execute the op.
       2.2. 'attrs_t': Specifies the corresponding type of the attribute. Can be empty. If
                       specified, the attrs argument will be downcast to this type and stored in
                       `casted_ptr`.
       2.3. 'dlts': A code snippet which specifies how to extract information from the `args`.
       2.4. 'order': Can be either a (condition, [order true], [order false]) tuple, or a [order]
                     list, which specifies how MNM tensors correspond to cuDNN tensor descriptors
                     and pointers.
       2.3. [var name]: The specific rule of generating this variable.
            2.3.1. str: just dump this plain string to replace this var name.
            2.3.2. dict, [dict]: We need to call other cuDNN APIs to prepare this parameter. This
                                 is specific to several data types:
                   2.3.2.1. xxDescriptor_t: It will first call the `Create` API, and then use rules
                                            specified in this field to `Set` the descriptor.
                   2.3.2.2. xxAlgo_t: Because algorithm-find APIs are often very slow, the results
                                      will be cached in a k-v map. By default, the key is the shapes
                                      of the input/output tensor. Sometimes, it is not enough. Thus,
                                      `extrakeys` fields can be specified so that these fields will
                                      also be combined in the key.
                                      To make our codegen life easier, it will generate a wrapper
                                      (helper) function to call the API and cache the results.
                   2.3.2.3. xx[SizeInBytes]: `xx` is the prefix of a corresponding xxSizeInBytes.
                            2.3.2.3.1. Sometimes `xx` does not exactly match `xxSizeInBytes` so
                                       'size_t' can be specified to find the corresponding argument.
                            2.3.2.3.2. Sometimes these size and pointer are shared through forward
                                       and backward so a 'res' can be provided to store this in a
                                       OpaqueValue.
                   After the generation above is done, these value will be replaced by the `str`
                   to follow the rule 2.3.1.
"""

#TODO(@were): Merge similar configurations together.
rules = [
('add', {
    'callee' : 'cudnnAddTensor',
    'cDesc'  : 'aDesc',
}),
('conv2d', {
    'attrs_t'   : 'ConvAttrs',
    'callee'    : 'cudnnConvolutionForward',
    'algo'      : {'callee': None,
                   'extrakeys': ['casted_ptr->padding',
                                 'casted_ptr->stride',
                                 'casted_ptr->dilation',]},
    'convDesc'  : [{
                     'callee'        : 'cudnnSetConvolutionNdDescriptor',
                     'arrayLength'   : 'casted_ptr->stride.size()',
                     'padA'          : TO_INT_PTR % 'casted_ptr->padding',
                     'filterStrideA' : TO_INT_PTR % 'casted_ptr->stride',
                     'dilationA'     : TO_INT_PTR % 'casted_ptr->dilation',
                     'mode'          : 'CUDNN_CROSS_CORRELATION',},
                   {'callee'     : 'cudnnSetConvolutionGroupCount',
                    'groupCount' : 'casted_ptr->groups',}],
    'workSpace' : {'workspace': None,},
}),
('grad.conv2d_data', {
    'callee'    : 'cudnnConvolutionBackwardData',
    'attrs_t'   : 'ConvBackAttrs',
    'algo'      : {'callee': None,
                   'extrakeys': ['casted_ptr->padding',
                                 'casted_ptr->stride',
                                 'casted_ptr->dilation']},
    'convDesc'  : [{
                     'callee'        : 'cudnnSetConvolutionNdDescriptor',
                     'arrayLength'   : 'casted_ptr->stride.size()',
                     'padA'          : TO_INT_PTR % 'casted_ptr->padding',
                     'filterStrideA' : TO_INT_PTR % 'casted_ptr->stride',
                     'dilationA'     : TO_INT_PTR % 'casted_ptr->dilation',
                     'mode'          : 'CUDNN_CROSS_CORRELATION',},
                   {'callee'     : 'cudnnSetConvolutionGroupCount',
                    'groupCount' : 'casted_ptr->groups',}],
    'workSpace' : [{'workspace': None,}]
}),
('grad.conv2d_filter', {
    'callee'    : 'cudnnConvolutionBackwardFilter',
    'attrs_t'   : 'ConvAttrs',
    'algo'      : {'callee': None,
                   'extrakeys': ['casted_ptr->padding',
                                 'casted_ptr->stride',
                                 'casted_ptr->dilation',]},
    'convDesc'  : [{
                     'callee'        : 'cudnnSetConvolutionNdDescriptor',
                     'arrayLength'   : 'casted_ptr->stride.size()',
                     'padA'          : TO_INT_PTR % 'casted_ptr->padding',
                     'filterStrideA' : TO_INT_PTR % 'casted_ptr->stride',
                     'dilationA'     : TO_INT_PTR % 'casted_ptr->dilation',
                     'mode'          : 'CUDNN_CROSS_CORRELATION',},
                   {'callee'     : 'cudnnSetConvolutionGroupCount',
                    'groupCount' : 'casted_ptr->groups',}],
    'workSpace' : [{'workspace'  : None,
                    'gradDesc': 'dwDesc'}]
}),
('relu', {
    'callee'         : 'cudnnActivationForward',
    'activationDesc' : {'callee' : 'cudnnSetActivationDescriptor',
                        'coef'   : '0.0',
                        'mode'   : 'CUDNN_ACTIVATION_RELU',}
}),
('grad.relu', {
    'callee'         : 'cudnnActivationBackward',
    'activationDesc' : {'callee' : 'cudnnSetActivationDescriptor',
                        'coef'   : '0.0',
                        'mode'   : 'CUDNN_ACTIVATION_RELU',}
}),
('tanh', {
    'callee'         : 'cudnnActivationForward',
    'activationDesc' : {'callee' : 'cudnnSetActivationDescriptor',
                        'coef'   : '0.0',
                        'mode'   : 'CUDNN_ACTIVATION_TANH',}
}),
('grad.tanh', {
    'callee'         : 'cudnnActivationBackward',
    'activationDesc' : {'callee' : 'cudnnSetActivationDescriptor',
                        'coef'   : '0.0',
                        'mode'   : 'CUDNN_ACTIVATION_TANH',}
}),
('sigmoid', {
    'callee'         : 'cudnnActivationForward',
    'activationDesc' : {'callee' : 'cudnnSetActivationDescriptor',
                        'coef'   : '0.0',
                        'mode'   : 'CUDNN_ACTIVATION_SIGMOID',}
}),
('grad.sigmoid', {
    'callee'         : 'cudnnActivationBackward',
    'activationDesc' : {'callee' : 'cudnnSetActivationDescriptor',
                        'coef'   : '0.0',
                        'mode'   : 'CUDNN_ACTIVATION_SIGMOID',}
}),
('softmax', OrderedDict(
    callee     = 'cudnnSoftmaxForward',
    attrs_t    = 'SoftmaxAttrs',
    init_extra = """
    int axis = casted_ptr->axis;
    int left = 1, center = dlts[0]->shape[axis], right = 1;
    for (int i = 0; i < axis; ++i) {
      left *= dlts[0]->shape[i];
    }
    for (int i = axis + 1; i < dlts[0]->ndim; ++i) {
      right *= dlts[0]->shape[i];
    }
""",
    shapes     = {
        0: 'left, center, right, 1',
        1: 'left, center, right, 1',
    },
    algo       = 'CUDNN_SOFTMAX_ACCURATE',
    mode       = ('center == 1 && right == 1', 'CUDNN_SOFTMAX_MODE_INSTANCE', 'CUDNN_SOFTMAX_MODE_CHANNEL'),
)),
('log_softmax', OrderedDict(
    callee     = 'cudnnSoftmaxForward',
    attrs_t    = 'SoftmaxAttrs',
    init_extra = """
    int axis = casted_ptr->axis;
    int left = 1, center = dlts[0]->shape[axis], right = 1;
    for (int i = 0; i < axis; ++i) {
      left *= dlts[0]->shape[i];
    }
    for (int i = axis + 1; i < dlts[0]->ndim; ++i) {
      right *= dlts[0]->shape[i];
    }
""",
    shapes     = {
        0: 'left, center, right, 1',
        1: 'left, center, right, 1',
    },
    algo       = 'CUDNN_SOFTMAX_LOG',
    mode       = ('center == 1 && right == 1', 'CUDNN_SOFTMAX_MODE_INSTANCE', 'CUDNN_SOFTMAX_MODE_CHANNEL'),
)),
('grad.softmax', OrderedDict(
    callee     = 'cudnnSoftmaxBackward',
    attrs_t    = 'SoftmaxAttrs',
    init_extra = """
    int axis = casted_ptr->axis;
    int left = 1, center = dlts[0]->shape[axis], right = 1;
    for (int i = 0; i < axis; ++i) {
      left *= dlts[0]->shape[i];
    }
    for (int i = axis + 1; i < dlts[0]->ndim; ++i) {
      right *= dlts[0]->shape[i];
    }
""",
    shapes     = {
        0: 'left, center, right, 1',
        1: 'left, center, right, 1',
    },
    algo       = 'CUDNN_SOFTMAX_ACCURATE',
    mode       = ('center == 1 && right == 1', 'CUDNN_SOFTMAX_MODE_INSTANCE', 'CUDNN_SOFTMAX_MODE_CHANNEL'),
)),
('grad.log_softmax', OrderedDict(
    callee     = 'cudnnSoftmaxBackward',
    attrs_t    = 'SoftmaxAttrs',
    init_extra = """
    int axis = casted_ptr->axis;
    int left = 1, center = dlts[0]->shape[axis], right = 1;
    for (int i = 0; i < axis; ++i) {
      left *= dlts[0]->shape[i];
    }
    for (int i = axis + 1; i < dlts[0]->ndim; ++i) {
      right *= dlts[0]->shape[i];
    }
""",
    shapes     = {
        0: 'left, center, right, 1',
        1: 'left, center, right, 1',
    },
    algo       = 'CUDNN_SOFTMAX_ACCURATE',
    mode       = ('center == 1 && right == 1', 'CUDNN_SOFTMAX_MODE_INSTANCE', 'CUDNN_SOFTMAX_MODE_CHANNEL'),
)),
('max_pool2d', {
    'callee'     : 'cudnnPoolingForward',
    'attrs_t'    : 'MaxPoolAttrs',
    'poolingDesc': {'callee'     : 'cudnnSetPoolingNdDescriptor',
                    'nbDims'     : 'casted_ptr->kernel.size()',
                    'windowDimA' : TO_INT_PTR % 'casted_ptr->kernel',
                    'paddingA'   : TO_INT_PTR % 'casted_ptr->padding',
                    'strideA'    : TO_INT_PTR % 'casted_ptr->stride',
                    'mode'       : 'CUDNN_POOLING_MAX',}
}),
('grad.max_pool2d', {
    'callee'     : 'cudnnPoolingBackward',
    'attrs_t'    : 'MaxPoolAttrs',
    'poolingDesc': {'callee'     : 'cudnnSetPoolingNdDescriptor',
                    'nbDims'     : 'casted_ptr->kernel.size()',
                    'windowDimA' : TO_INT_PTR % 'casted_ptr->kernel',
                    'paddingA'   : TO_INT_PTR % 'casted_ptr->padding',
                    'strideA'    : TO_INT_PTR % 'casted_ptr->stride',
                    'mode'       : 'CUDNN_POOLING_MAX',}
}),
('avg_pool2d', {
    'callee'     : 'cudnnPoolingForward',
    'attrs_t'    : 'AvgPoolAttrs',
    'poolingDesc': {'callee'     : 'cudnnSetPoolingNdDescriptor',
                    'nbDims'     : 'casted_ptr->kernel.size()',
                    'windowDimA' : TO_INT_PTR % 'casted_ptr->kernel',
                    'paddingA'   : TO_INT_PTR % 'casted_ptr->padding',
                    'strideA'    : TO_INT_PTR % 'casted_ptr->stride',
                    'mode'       : ('casted_ptr->include_pad',
                                    'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING',
                                    'CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING',),}
}),
('grad.avg_pool2d', {
    'callee'     : 'cudnnPoolingBackward',
    'attrs_t'    : 'AvgPoolAttrs',
    'poolingDesc': {'callee'     : 'cudnnSetPoolingNdDescriptor',
                    'nbDims'     : 'casted_ptr->kernel.size()',
                    'windowDimA' : TO_INT_PTR % 'casted_ptr->kernel',
                    'paddingA'   : TO_INT_PTR % 'casted_ptr->padding',
                    'strideA'    : TO_INT_PTR % 'casted_ptr->stride',
                    'mode'       : ('casted_ptr->include_pad',
                                    'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING',
                                    'CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING',),}
}),
('dropout', {
    'dlts'                    : """
    int n = args.size();
    std::vector<const DLTensor*> dlts(n);
    for (int i = 0; i < n - 1; ++i) {
      dlts[i] = args[i];
    }
    value::TupleValue tv = Downcast<TupleValue>(args[n - 1]);
    dlts[n - 1] = tv->fields[0];
    std::vector<OpaqueValue> opqs{Downcast<OpaqueValue>(tv->fields[1]),
                                  Downcast<OpaqueValue>(tv->fields[2])};
""",
    'callee'       : 'cudnnDropoutForward',
    'attrs_t'      : 'DropoutAttrs',
    'dropoutDesc'  : {'callee'  : 'cudnnSetDropoutDescriptor',
                      'dropout' : 'casted_ptr->dropout',
                      'seed'    : 'casted_ptr->seed',
                      'states'  : {'reserve' : 'cudnnDropoutGetStatesSize',
                                   'res'     : 'opqs[0]',
                                   'size_t'  : 'stateSizeInBytes'},},
    'reserveSpace' : {'reserve' : 'cudnnDropoutGetReserveSpaceSize',
                      'res'     : 'opqs[1]'},
}),
('grad.dropout', {
    'dlts'                    : """
    int n = args.size();
    std::vector<const DLTensor*> dlts(n);
    for (int i = 0; i < n - 1; ++i) {
      dlts[i] = args[i];
    }
    value::TupleValue tv = Downcast<TupleValue>(args[n - 1]);
    dlts[n - 1] = tv->fields[0];
    std::vector<OpaqueValue> opqs{Downcast<OpaqueValue>(tv->fields[1]),
                                  Downcast<OpaqueValue>(tv->fields[2])};
""",
    'callee'                  : 'cudnnDropoutBackward',
    'attrs_t'                 : 'DropoutAttrs',
    'dropoutDesc'             : {'callee'           : 'cudnnRestoreDropoutDescriptor',
                                 'dropout'          : 'casted_ptr->dropout',
                                 'seed'             : 'casted_ptr->seed',
                                 'states'           : 'Downcast<BufferValue>(opqs[0])->data',
                                 'stateSizeInBytes' : 'Downcast<BufferValue>(opqs[0])->size_in_bytes'},
    'reserveSpace'            : 'Downcast<BufferValue>(opqs[1])->data',
    'reserveSpaceSizeInBytes' : 'Downcast<BufferValue>(opqs[1])->size_in_bytes',
}),
('batch_norm2d', (
{
    'attrs_t' : 'BatchNormAttrs',
    'cond'    : 'casted_ptr->is_training'
},
{
    'callee'                   : 'cudnnBatchNormalizationForwardTraining',
    'attrs_t'                  : 'BatchNormAttrs',
    'order'                    : [0, 5, 1, 2, 3, 4],
    'shapes'                   : { 1: '1, (int) dlts[1]->shape[0], 1, 1', },
    'mode'                     : 'CUDNN_BATCHNORM_SPATIAL',
    'exponentialAverageFactor' : 'casted_ptr->momentum',
    'epsilon'                  : 'casted_ptr->eps',
    'resultRunningMean'        : 'DLTensor:dlts[1]',
    'resultRunningVariance'    : 'DLTensor:dlts[2]',
    'bnScale'                  : 'DLTensor:dlts[3]',
    'bnBias'                   : 'DLTensor:dlts[4]',
    'resultSaveMean'           : 'nullptr',
    'resultSaveInvVariance'    : 'nullptr',
},
{
    'callee'                   : 'cudnnBatchNormalizationForwardInference',
    'attrs_t'                  : 'BatchNormAttrs',
    'order'                    : [0, 5, 1, 2, 3, 4],
    'shapes'                   : { 1: '1, (int) dlts[1]->shape[0], 1, 1', },
    'mode'                     : 'CUDNN_BATCHNORM_SPATIAL',
    'exponentialAverageFactor' : 'casted_ptr->momentum',
    'epsilon'                  : 'casted_ptr->eps',
    'estimatedMean'            : 'DLTensor:dlts[1]',
    'estimatedVariance'        : 'DLTensor:dlts[2]',
    'bnScale'                  : 'DLTensor:dlts[3]',
    'bnBias'                   : 'DLTensor:dlts[4]',
})),
('grad.batch_norm2d', {
    'dlts'             : """
    int n = args.size();
    std::vector<const DLTensor*> dlts(n + 3);
    for (int i = 0; i < n; ++i) {
      dlts[i] = args[i];
    }
    value::TupleValue tv = Downcast<TupleValue>(info->output);
    for (int i = 0; i < 3; ++i) {
      dlts[n + i] = tv->fields[i];
    }
""",
    'callee'           : 'cudnnBatchNormalizationBackward',
    'attrs_t'          : 'BatchNormAttrs',
    'order'            : [0, 1, 3, 2, 4, 5],
    'mode'             : 'CUDNN_BATCHNORM_SPATIAL',
    'alphaDataDiff'    : 'CUDNNDType(dtype).const_addr<1>()',
    'betaDataDiff'     : 'CUDNNDType(dtype).const_addr<0>()',
    'alphaParamDiff'   : 'CUDNNDType(dtype).const_addr<1>()',
    'betaParamDiff'    : 'CUDNNDType(dtype).const_addr<0>()',
    'shapes'           : { 2: '1, (int) dlts[2]->shape[0], 1, 1' },
    'dBnScaleResult'   : 'DLTensor:dlts[4]',
    'dBnBiasResult'    : 'DLTensor:dlts[5]',
    'bnScale'          : 'DLTensor:dlts[2]',
    'savedMean'        : 'nullptr',
    'savedInvVariance' : 'nullptr',
    'epsilon'          : 'casted_ptr->eps',
})
]

import emitter
emit = emitter.Emitter(file_name, funcs)

emit.emit_openv(rules)

print('Unused APIs:')
f = lambda i: i not in emit.func_used and ('Backward' in i or 'Forward' in i) and \
        ('Get' not in i and 'Find' not in i)
for num, func in enumerate([i for i in emit.cudnn_api.keys() if f(i)]):
    print(num, func)

del emit

shutil.move('%s.cc' % file_name, '../../src/op/backend/cudnn/%s.cc' % file_name)
subprocess.check_output(['../clang-format.sh', '../../src/op/backend/cudnn/%s.cc' % file_name])
