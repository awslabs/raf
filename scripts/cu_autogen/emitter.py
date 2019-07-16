import utils
import os

# TODO(@were): Auto-generate test cases.

class Emitter(object):

    namespaces = ['mnm', 'op', 'backend', 'cudnn', 'generated']
    headers = [
            '#include <cudnn.h>', '#include <mnm/op.h>',
            '#include "./util.h"',
            '#include "../../../common/arg_utils.h"',
            '#include "../../../common/shape_utils.h"',
            '#include "../../../common/cuda.h"'
            ]

    finder_fmt = """%s %s(const std::vector<int> &key, %s) {
  if (%s.has(key)) {
    return %s.get(key);
  }
  int cnt;
  %s res;
  %s
  if (res.status != CUDNN_STATUS_SUCCESS) {
    LOG(FATAL) << "ValueError: Cannot find a proper algorithm!\\n";
    throw;
  }
  %s.set(key, res.algo);
  return res.algo;
}
"""
    args_attrs = 'rly::Array<value::Value> args, rly::Attrs attrs'
    desc_t = 'Descriptor_t'
    prfx = 'cudnn'

    def __init__(self, prefix, parser_info):
        self._cc = open('%s.cc' % prefix, 'w')
        self.indent_ = 0

        self.cudnn_api = {}
        self.cudnn_api = parser_info.parsed_funcs
        self.cudnn_enum = parser_info.parsed_enums

        self.needed_enum = set()
        self.extra_fields = []
        self.extra_fields_by_key = {}
        self.extra_wrappers = []

        self.default = {
            'cudnnDataType_t'       : 'CUDNNDType(dtype)',
            'cudnnNanPropagation_t' : 'CUDNN_PROPAGATE_NAN',
            'cudnnHandle_t'         : 'CUDNNThreadEntry::ThreadLocal()->handle',
        }

        self.write('\n'.join(Emitter.headers) + '\n')
        for fs in os.listdir('../../src/op/attrs'):
            if fs.endswith('.h'):
                self.write('#include "../../attrs/%s"' % fs)
        for ns in Emitter.namespaces:
            self.write('namespace %s {' % ns)
        self.write('\n')


    def __del__(self):
        self.write('\n')
        for ns in Emitter.namespaces[::-1]:
            self.write('} // namespaces %s' % ns)


    def write(self, s):
        if s == '\n':
            self._cc.write('\n')
            return
        self._cc.write((self.indent_ * ' ' + '%s\n') % s)


    def indent(self, delta):
        self.indent_ += delta


    def emit_enum(self):
        class_fmt = 'class %s : public EnumBase<%s, %d, int32_t, %s> {\n public:'
        enum_def_fmt = 'ENUM_DEF_ENTRY_WITH_NAME(%s, %d, %s, %s, "%s");'

        # Skip this for now
        self.cudnn_enum.pop('cudnnFusedOpsVariantParamLabel_t')
        # This is handled by special rule
        self.cudnn_enum.pop('cudnnDataType_t')

        for name in self.needed_enum:
            elems = self.cudnn_enum[name]
            wrapper_name = '%sEnum' % name[5:-2]
            self.write(class_fmt % (wrapper_name, wrapper_name, len(elems), name))
            self.indent(2)
            self.write('ENUM_DEF_HEADER(%s, 0, plain);' % (wrapper_name))
            for elem in elems:
                s, v = elem
                self.write(enum_def_fmt % (wrapper_name, v, utils.format_enum_name(s), s, s))
            self.indent(-2)
            self.write('};\n')


    def emit_openv(self, rules):
        # Dump it twice
        # The first time is to figure out which enums to dump
        old = self._cc
        self._cc = open('/dev/null', 'w')
        for op, rule in rules:
            _rule = rule.copy()
            self._emit_openv(op, _rule)
            self.extra_fields_by_key[op] = self.extra_fields
            self.extra_fields = []

        self._cc = old
        self.emit_enum()
        self.write('\n'.join(self.extra_wrappers))
        for op, rule in rules:
            self.extra_fields = self.extra_fields_by_key[op]
            self._emit_openv(op, rule)


    def _emit_preallocate(self, op_name, op_rule):
        args = self.cudnn_api[op_rule['callee']]

        # Emit constructor
        self.write('void PreAllocate(%s) {' % self.args_attrs)
        self.indent(2)
        self.write('auto dlts = common::arg_utils::AsVector(args);')
        self.write('dtype = common::arg_utils::DeduceDLType(dlts);')
        self.write('auto ctx = common::arg_utils::DeduceCtx(dlts);')
        if 'attrs_t' in op_rule.keys():
            self.write('static auto casted_ptr = attrs.as<attrs::%s>();' % op_rule['attrs_t'])
        self.write('\n')

        last_tensor_desc = None
        cur = 0
        order = op_rule.get('order', list(range(len(args))))
        shapes = []
        strides = []
        for arg in args:
            if arg.type.endswith(Emitter.desc_t):
                func = 'cudnnCreate%sDescriptor' % arg.type[len(Emitter.prfx):-len(Emitter.desc_t)]
                assert func in self.cudnn_api.keys()
                self.write('CUDNN_CALL(%s(&%s));' % (func, arg.name))
                if arg.type in ['cudnnTensorDescriptor_t', 'cudnnFilterDescriptor_t']:
                    idx = order[cur]
                    self.write('FORM_SHAPE(shape_%d, dlts[%d]);' % (idx, idx))
                    shape = 'shape_%d' % cur
                    shapes.append(shape)
                    if arg.type == 'cudnnTensorDescriptor_t':
                        self.write('FORM_STRIDE(stride_%d, shape_%d);' % (idx, idx))
                        stride = 'stride_%d' % idx
                        strides.append(stride)
                        tensor_args = [arg.name,
                                       None,
                                       '%s.size()' % shape,
                                       'dmlc::BeginPtr(%s)' % shape,
                                       'dmlc::BeginPtr(%s)' % stride]
                    else:
                        tensor_args = [arg.name,
                                       None,
                                       'CUDNN_TENSOR_NCHW', # TODO(@were): should I hard code this?
                                       '%s.size()' % shape,
                                       'dmlc::BeginPtr(%s)' % shape]
                    func_name = 'cudnnSet' + arg.type[5:-len(Emitter.desc_t)] + 'NdDescriptor'
                    self.emit_func_call(func_name,
                                        tensor_args,
                                        op_rule)
                    last_tensor_desc = arg
                    cur += 1
                elif arg.name in op_rule.keys():
                    to_do = op_rule[arg.name]
                    to_do = to_do if isinstance(to_do, list) else [to_do]
                    for elem in to_do:
                        assert 'Set' in elem
                        assert elem in self.cudnn_api
                        callee_args = [arg.name] + [None] * (len(self.cudnn_api[elem]) - 1)
                        self.emit_func_call(elem, callee_args, op_rule)
                self.write('\n')

            if last_tensor_desc is not None:
                is_prfx = last_tensor_desc.name.startswith(arg.name)
                is_prfx = is_prfx or last_tensor_desc.name.startswith(arg.name.lower())
            else:
                is_prfx = False
            if arg.__str__().startswith('void *') and is_prfx:
                num = order[cur - 1]
                self.write('int size_%d = ((shape_%d[0] * stride_%d[0]) * dtype.bits - 1) / 8 + 1;' % (num, num, num))
                self.write('RequestMemory(const_cast<void**>(&dlts[%d]->data), ctx, size_%d);' % (num, num))
                self.write('\n')

        for arg in args:
            if arg.type.endswith(Emitter.desc_t):
                continue

            if arg.is_attr_field() and arg.name in op_rule.keys():
                if arg.type.endswith('Algo_t'):
                    finder, param_list = op_rule[arg.name]
                    self.write('auto key = ConcatVecs(' + ', '.join(shapes + strides + param_list) + ');')
                    arg_list = self.cudnn_api[finder]
                    if not finder.endswith('Ex'):
                        perf_t = arg_list[-1].type
                        arg_list = arg_list[1:-3]
                        cache_name = '_cache_%s' % arg.type;
                        cache_decl = 'AlgorithmCache<%s> %s;' % (arg.type, cache_name)
                        arg_list_decl = ', '.join(i.__str__() for i in arg_list)
                        finder_call = self._emit_func_call(finder, [None] + [i.name for i in arg_list] + ['1', '&cnt', '&res'], op_rule)
                        finder_decl = Emitter.finder_fmt % (arg.type, finder[5:], arg_list_decl,
                                                            cache_name, cache_name, perf_t,
                                                            finder_call, cache_name)
                        self.extra_wrappers.append(cache_decl)
                        self.extra_wrappers.append(finder_decl)
                    else:
                        assert False, "Not supported yet!"
                    self.write('%s(key, %s);' % (finder[len(Emitter.prfx):], ', '.join(i.name for i in arg_list)))
                elif 'SizeInBytes' in arg.name:
                    self._emit_func_arg(arg, None, op_rule)
                    self.write('this->%s = %s;' % (arg.name, arg.name))
                elif isinstance(op_rule[arg.name], str):
                    self.write('%s = %s;' % (arg.name, self._emit_func_arg(arg, None, op_rule)))
                op_rule[arg.name] = arg.name

        self.indent(-2)
        self.write('}\n')


    def _emit_openv(self, op_name, op_rule):
        args = self.cudnn_api[op_rule['callee']]
        self.emit_func_comments(op_rule['callee'])

        class_fmt = 'class % final : public '
        class_name = '%s_for_op_%s' % (op_rule['callee'][5:], op_name)
        self.write('class %s : public mnm::op::OpEnv {\n public:\n' % class_name)

        self.indent(2)
        self.write('%s() {}\n' % class_name)

        # Emit op fileds
        self.write('DType dtype;')
        for arg in args:
            if arg.is_attr_field():
                self.write('%s %s%s;' % (arg.type, arg.is_ptr, arg.name))
        for arg in self.extra_fields:
            self.write(arg)
        self.write('\n')

        self._emit_preallocate(op_name, op_rule)

        # Emit destructor
        self.write('~%s() {' % class_name)
        self.indent(2)
        for arg in args:
            if arg.type.endswith('Descriptor_t'):
                func = 'cudnnDestroy%sDescriptor' % arg.type[len(Emitter.prfx):-len(Emitter.desc_t)]
                assert func in self.cudnn_api.keys()
                self.write('CUDNN_CALL(%s(%s));' % (func, arg.name))
        self.indent(-2)
        self.write('}\n')

        # Emit maker
        self.write('static OpEnv *make(%s) {' % Emitter.args_attrs)
        self.indent(2)
        self.write('std::unique_ptr<%s> res = std::make_unique<%s>();' % (class_name, class_name))
        self.write('res->PreAllocate(args, attrs);')
        self.write('return res.release();')
        self.indent(-2)
        self.write('}\n')

        # Emit execute
        self.write('void Execute(%s) override final {' % Emitter.args_attrs)
        self.indent(2)
        func_args = []
        cur = 0
        self.write('auto dlts = common::arg_utils::AsVector(args);')
        order = op_rule.get('order', list(range(len(args))))

        def in_extra_fields(lst, v):
            for elem in lst:
                if elem[:-1].endswith(v):
                    return True
            return False

        for arg in args:
            if arg.name == 'handle':
                func_args.append(None)
            elif arg.is_attr_field() or in_extra_fields(self.extra_fields, arg.name):
                func_args.append(arg.name)
            elif arg.is_const():
                func_args.append('CUDNNDType(dtype).const_addr<%d>()' % (1 if 'alpha' in arg.name else 0))
            elif 'void *' in arg.__str__():
                func_args.append('dlts[%d]->data' % order[cur])
                cur += 1
        self.emit_func_call(op_rule['callee'], func_args, op_rule)
        self.indent(-2)
        self.write('}\n')

        self.write('// TODO(@were): After executor is done, remove these two!')
        self.write('// TODO(@junrushao1994): Implement the executor!')
        self.write('void RequestMemory(void** dest, Context ctx, int64_t nb) { CUDA_CALL(cudaMalloc(dest, nb)); }')
        self.write('void RequestWorkspace(void** dest, Context ctx, int64_t nb) { CUDA_CALL(cudaMalloc(dest, nb)); }\n')

        self.indent(-2)
        self.write('};\n')

        reg_fmt = 'MNM_REGISTER_OP_DISPATCH("mnm.op.%s", DevType::kCUDA(), "generated_cudnn", %s::make);'
        self.write(reg_fmt % (op_name, class_name))
        self.write('\n\n')


    def _emit_func_arg(self, param, arg, op_rule):
        if arg is not None:
            res = arg
        elif param.name in op_rule.keys():
            res = op_rule[param.name]
            # This slightly violates my design philosophy of this function.
            # Originally, I want to make this function generate str only.
            # External caller will dump the string.
            if param.type == 'size_t' and param.name.endswith('SizeInBytes'):
                callee = res
                callee_args = []
                assert callee in self.cudnn_api.keys()
                for elem in self.cudnn_api[callee]:
                    if elem.type.endswith(Emitter.desc_t):
                        callee_args.append(elem.name)
                    elif elem.__str__() == 'size_t *sizeInBytes':
                        callee_args.append('&%s' % param.name)
                    else:
                        callee_args.append(None)
                self.write('size_t %s;' % param.name)
                self.emit_func_call(callee, callee_args, op_rule)
                if param.name.startswith('workSpace'):
                    self.write('RequestWorkspace(&workSpace, ctx, %s);' % param.name)
                    extra_filed = 'workSpace';
                else:
                    self.write('RequestMemory(&%s, ctx, %s);' % (param.name[:-len('SizeInBtyes')],
                                                                 param.name))
                    extra_filed = param.name[:-len('SizeInBtyes')];
                self.extra_fields.append('void *%s;' % extra_filed)
                op_rule.pop(param.name)
                op_rule[extra_filed] = extra_filed
                res = param.name
        elif param.type in self.default.keys():
            res = self.default[param.type]
        else:
            res = param.name

        if param.type in self.cudnn_enum.keys() and param.type != 'cudnnDataType_t':
            ripped = param.type[5:-2]
            replaces = []
            for i in res.split():
                if i.startswith('CUDNN_'):
                    replaces.append((i, '%sEnum(%sEnum::%s())' % (ripped, ripped,
                                                                  utils.format_enum_name(i))))
            for src, dst in replaces:
                res = res.replace(src, dst)
            self.needed_enum.add(param.type)

        return res


    def _emit_func_call(self, func, args, op_rule):
        indent = self.indent_ + len(func) + 2 + 10
        res = []
        assert len(self.cudnn_api[func]) == len(args)
        for param, arg in zip(self.cudnn_api[func], args):
            s = self._emit_func_arg(param, arg, op_rule)
            res.append(s)
        return 'CUDNN_CALL(' + func + '(' + (',\n' + indent * ' ').join(res) + '));'
    

    def emit_func_call(self, func, args, op_rule):
        self.write(self._emit_func_call(func, args, op_rule))


    def emit_func_comments(self, func):
        indent = self.indent_ + len(func) + 1
        args = [i.__str__() for i in self.cudnn_api[func]]
        self.write('// ' + func + '(' + (',\n// ' + indent * ' ').join(args) + ')')
