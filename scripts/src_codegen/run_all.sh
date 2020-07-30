#!/bin/bash
set -e
set -u
set -o pipefail
set -x

python3 scripts/src_codegen/main_cxx_schema.py
python3 scripts/src_codegen/main_cxx_reg.py && scripts/lint/run-clang-format.sh src/op/regs/regs.cc
python3 scripts/src_codegen/main_py_sym.py
python3 scripts/src_codegen/main_py_imp.py
python3 scripts/src_codegen/main_py_ffi.py
python3 scripts/src_codegen/main_backend_cudnn.py && scripts/lint/run-clang-format.sh src/op/dispatch/cudnn/impl.cc
# The script below requires libtvm.so, and therefore it is not mandatory
#   python3 scripts/src_codegen/extract_relay_ops.py > scripts/src_codegen/def_tvm_op.py
python3 scripts/src_codegen/main_cxx_tvm_op.py && scripts/lint/run-clang-format.sh src/op/regs/tvmjit_regs.cc
