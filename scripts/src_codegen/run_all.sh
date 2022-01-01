#!/bin/bash
set -e
set -u
set -o pipefail
set -x

python3 -m scripts.src_codegen.main_cxx_schema && scripts/lint/run-clang-format.sh src/op/schema/*
python3 -m scripts.src_codegen.main_cxx_reg && scripts/lint/run-clang-format.sh src/op/regs/regs.cc
python3 -m scripts.src_codegen.main_py_sym && black -l 100 python/mnm/_op/sym.py
python3 -m scripts.src_codegen.main_py_imp && black -l 100 python/mnm/_op/imp.py
python3 -m scripts.src_codegen.main_py_ir && black -l 100 python/mnm/ir/op.py
rm -r python/mnm/_ffi/* && python3 -m scripts.src_codegen.main_py_ffi && black -l 100 python/mnm/_ffi
# Disable the codegen of cudnn backend, now cudnn backend is manually written
# python3 -m scripts.src_codegen.main_backend_cudnn && scripts/lint/run-clang-format.sh src/op/dispatch/cudnn/impl.cc
# The script below requires libtvm.so, and therefore it is not mandatory
# TODO: python3 -m scripts.src_codegen.extract_relay_ops
python3 -m scripts.src_codegen.main_cxx_tvm_op && scripts/lint/run-clang-format.sh src/op/regs/tvm_op_regs.cc
