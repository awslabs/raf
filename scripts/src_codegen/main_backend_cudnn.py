import pycparser
import os
import sys
import shutil
import subprocess
import re


def preprocess(f_name, ind_dir):
    raw = subprocess.check_output(["gcc", "-E", f_name, ind_dir]).decode('utf-8')

    raw = re.sub('__attribute__\\(\\(.*\\)\\)', '', raw)
    raw = raw.replace('__host__', '')
    raw = raw.replace('__inline__', '')
    return raw


class CUDNNArg(object):

    def __init__(self, const, ty, is_ptr, name):
        self.const = const
        self.type = ty
        self.is_ptr = is_ptr
        self.name = name
        self.parent = None
        self.arg_id = None


    def __str__(self):
        return "{CONST}{TYPE} {PTR}{NAME}".format(CONST=self.const, TYPE=self.type, PTR=self.is_ptr,
                                                  NAME=self.name)


    def is_attr_field(self):
        # I am not sure if this is too adhoc
        if self.type == 'cudnnHandle_t':
            return False
        return self.type.endswith('_t') and self.type.startswith('cudnn')


def extract_param_info(node):
    class ParamVisitor(pycparser.c_ast.NodeVisitor):

        def __init__(self):
            self.type_name = None
            self.var_name = None
            self.is_ptr = ''
            self.is_const = ''


        def visit_IdentifierType(self, node):
            assert self.type_name is None
            self.type_name = ' '.join(node.names)


        def visit_TypeDecl(self, node):
            self.visit(node.type)
            if u'const' in node.quals:
                self.is_const = 'const '


        def visit_PtrDecl(self, node):
            self.is_ptr = self.is_ptr + '*'
            self.visit(node.type)
            if u'const' in node.quals:
                self.is_const = 'const '


        def visit_Decl(self, node):
            self.var_name = str(node.name)
            self.visit(node.type)

    pv = ParamVisitor()
    pv.visit(node)
    return CUDNNArg(pv.is_const, pv.type_name, pv.is_ptr, pv.var_name)


class ExtractVisitor(pycparser.c_ast.NodeVisitor):

    def __init__(self):
        self.parsed_funcs = {}


    def visit_Decl(self, node):
        if node.name and node.name.startswith('cudnn'):
            if isinstance(node.type, pycparser.c_ast.FuncDecl):
                if isinstance(node.type.args, pycparser.c_ast.ParamList):
                    func_name = str(node.name)
                    args = self.parsed_funcs[func_name] = []
                    for i in node.type.args:
                        arg = extract_param_info(i)
                        arg.parent = func_name
                        arg.arg_id = len(args)
                        args.append(arg)


def extract_functions(src):
    parser = pycparser.CParser()
    ast = parser.parse(src)
    visitor = ExtractVisitor()
    visitor.visit(ast)
    return visitor.parsed_funcs


def main(path='src/op/dispatch/cudnn/impl.cc'):
    cudnn_home = os.getenv("CUDNN_HOME")
    cuda_home = os.getenv("CUDA_HOME")
    cudnn_h = "%s/include/cudnn.h" % cudnn_home
    src = preprocess(cudnn_h, "-I%s/include" % cuda_home)
    cudnn_apis = extract_functions(src)

    import def_op
    ops = def_op.by_name()

    import def_schema
    schema = def_schema.by_name()

    import def_cudnn
    wrappers = dict()
    classes = [elem.normalize(ops, schema, cudnn_apis, wrappers) for elem in sorted(def_cudnn.SCHEMAS, key=lambda x:x.op)]
    classes = '\n\n'.join(classes)

    fmt = """
/*!
 * Copyright (c) 2020 by Contributors
 * Auto generated. Do not touch.
 * \\file {FILENAME}
 * \\brief Operator schema.
 */
{HEADERS}

namespace mnm {{
namespace op {{
namespace cudnn {{
namespace generated {{

using value::TupleValueObj;
using common::shape_utils::BytesCompactTensor;
using common::shape_utils::GetShape;
using common::shape_utils::PadDims;
using common::shape_utils::Shape2Strides;
using dmlc::BeginPtr;

{WRAPPERS}

{CLASSES}
}}  // namespace generated
}}  // namespace cudnn
}}  // namespace op
}}  // namespace mnm
""".strip()
    headers = [f'#include "../../schema/{i}"'
               for i in os.listdir('src/op/schema/') if i.endswith('.h')]
    headers += ['#include "./cudnn_utils.h"']
    headers += ['#include "../../op_utils.h"']
    headers = '\n'.join(sorted(headers))
    wrappers = '\n\n'.join(sorted(wrappers.values()))
    open(path, 'w').write(fmt.format(FILENAME=path, HEADERS=headers, CLASSES=classes, WRAPPERS=wrappers) + "\n")
    subprocess.check_output(['clang-format', '-i', path])

if __name__ == '__main__':
    main()
