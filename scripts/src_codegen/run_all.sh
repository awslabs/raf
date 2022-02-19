#!/bin/bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
# The script below requires libtvm.so, and therefore it is not mandatory
# TODO: python3 -m scripts.src_codegen.extract_relay_ops
python3 -m scripts.src_codegen.main_cxx_tvm_op && scripts/lint/run-clang-format.sh src/op/regs/tvm_op_regs.cc
