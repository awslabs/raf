import pycparser
import os
import sys
import shutil


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

TO_INT_PTR = 'dmlc::BeginPtr(common::shape_utils::MakeShape<int>(%s))'
TO_VEC = 'common::shape_utils::MakeShape<int>(%s)'

rules = [
('conv2d', {
    'callee'               : 'cudnnConvolutionForward',
    'attrs_t'              : 'Conv2DAttrs',
    'algo'                 : ('cudnnFindConvolutionForwardAlgorithm', ['casted_ptr->padding',
                                                                       'casted_ptr->stride',
                                                                       'casted_ptr->dilation']),
    'arrayLength'          : 'casted_ptr->stride.size()',
    'padA'                 : TO_INT_PTR % 'casted_ptr->padding',
    'filterStrideA'        : TO_INT_PTR % 'casted_ptr->stride',
    'dilationA'            : TO_INT_PTR % 'casted_ptr->dilation',
    'mode'                 : 'CUDNN_CROSS_CORRELATION',
    'convDesc'             : ['cudnnSetConvolutionNdDescriptor', 'cudnnSetConvolutionGroupCount'],
    'groupCount'           : 'casted_ptr->groups',
    'workSpaceSizeInBytes' : 'cudnnGetConvolutionForwardWorkspaceSize',
}),
('add', {
    'callee'         : 'cudnnAddTensor',
}),
('relu', {
    'callee'         : 'cudnnActivationForward',
    'activationDesc' : 'cudnnSetActivationDescriptor',
    'coef'           : '0.0',
    'mode'           : 'CUDNN_ACTIVATION_RELU',
}),
('tanh', {
    'callee'         : 'cudnnActivationForward',
    'activationDesc' : 'cudnnSetActivationDescriptor',
    'coef'           : '0.0',
    'mode'           : 'CUDNN_ACTIVATION_TANH',
}),
('sigmoid', {
    'callee'         : 'cudnnActivationForward',
    'activationDesc' : 'cudnnSetActivationDescriptor',
    'coef'           : '0.0',
    'mode'           : 'CUDNN_ACTIVATION_SIGMOID',
}),
('softmax', {
    'callee' : 'cudnnSoftmaxForward',
    'algo'   : 'CUDNN_SOFTMAX_ACCURATE',
    'mode'   : 'CUDNN_SOFTMAX_MODE_INSTANCE',
}),
('max_pool2d', {
    'callee'     : 'cudnnPoolingForward',
    'attrs_t'    : 'MaxPoolAttrs',
    'poolingDesc': 'cudnnSetPoolingNdDescriptor',
    'nbDims'     : 'casted_ptr->kernel_size.size()',
    'windowDimA' : TO_INT_PTR % 'casted_ptr->kernel_size',
    'paddingA'   : TO_INT_PTR % 'casted_ptr->padding',
    'strideA'    : TO_INT_PTR % 'casted_ptr->stride',
    'mode'       : 'CUDNN_POOLING_MAX',
}),
('avg_pool2d', {
    'callee'     : 'cudnnPoolingForward',
    'attrs_t'    : 'AvgPoolAttrs',
    'poolingDesc': 'cudnnSetPoolingNdDescriptor',
    'nbDims'     : 'casted_ptr->kernel_size.size()',
    'windowDimA' : TO_INT_PTR % 'casted_ptr->kernel_size',
    'paddingA'   : TO_INT_PTR % 'casted_ptr->padding',
    'strideA'    : TO_INT_PTR % 'casted_ptr->stride',
    'mode'       : 'casted_ptr->include_pad ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING',
}),
('dropout', {
    'callee'                  : 'cudnnDropoutForward',
    'attrs_t'                 : 'DropoutAttrs',
    'dropoutDesc'             : 'cudnnSetDropoutDescriptor',
    'reserveSpaceSizeInBytes' : 'cudnnDropoutGetReserveSpaceSize',
    'stateSizeInBytes'        : 'cudnnDropoutGetStatesSize',
    'dropout'                 : 'casted_ptr->dropout',
    'seed'                    : 'time(0)',
    'states'                  : 'state',
})
]

import emitter
emit = emitter.Emitter(file_name, funcs)

emit.emit_openv(rules)

del emit

shutil.copy2('%s.cc' % file_name, '../../src/op/backend/cudnn/')
