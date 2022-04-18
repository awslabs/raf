#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -u
set -o pipefail
set -x

python3 -m scripts.src_codegen.main_cxx_schema
python3 -m scripts.src_codegen.main_cxx_reg
python3 -m scripts.src_codegen.main_py_sym
python3 -m scripts.src_codegen.main_py_imp
python3 -m scripts.src_codegen.main_py_ir
rm -rf python/raf/_ffi && mkdir python/raf/_ffi && python3 -m scripts.src_codegen.main_py_ffi
python3 -m scripts.src_codegen.main_cxx_tvm_op
