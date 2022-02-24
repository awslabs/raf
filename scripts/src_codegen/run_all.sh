#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail
set -x

python3 -m scripts.src_codegen.main_cxx_schema && scripts/lint/run-clang-format.sh src/op/schema/*
python3 -m scripts.src_codegen.main_cxx_reg && scripts/lint/run-clang-format.sh src/op/regs/regs.cc
python3 -m scripts.src_codegen.main_py_sym && black -l 100 python/raf/_op/sym.py
python3 -m scripts.src_codegen.main_py_imp && black -l 100 python/raf/_op/imp.py
python3 -m scripts.src_codegen.main_py_ir && black -l 100 python/raf/ir/op.py
rm -rf python/raf/_ffi && mkdir python/raf/_ffi && python3 -m scripts.src_codegen.main_py_ffi && black -l 100 python/raf/_ffi
python3 -m scripts.src_codegen.main_cxx_tvm_op && scripts/lint/run-clang-format.sh src/op/regs/tvm_op_regs.cc
