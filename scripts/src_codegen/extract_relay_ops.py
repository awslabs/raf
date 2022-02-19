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

import sys

import tvm
from tvm import relay

PREFIX_BLACK_LIST = {
    "annotation.",
    "memory.",
    "qnn.",
    "vision.",
    "nn.",
    "contrib.",
    "_contrib_reverse_reshape",
}

BLACK_LIST = {
    "on_device",
    "device_copy",
    "relay.op.annotation.simulated_quantize",
}

WHILTE_LIST = {
    "nn.bias_add",
}

MNM_OP_NAME = {
    "nn.bias_add": "mnm.op.bias_add",
}


def collect_op():
    pattern_map = {
        0: "kElemWise",
        1: "kBroadcast",
        2: "kInjective",
        3: "kCommReduce",
        4: "kOutEWiseFusable",
        7: "kTuple",
        8: "kOpaque",
    }
    list_op = tvm.get_global_func("relay.op._ListOpNames")
    get_op = tvm.get_global_func("relay.op._GetOp")

    def is_black_listed(op_name):
        if op_name.startswith("mnm."):
            return True
        if op_name in WHILTE_LIST:
            assert op_name not in BLACK_LIST
            return False
        if op_name in BLACK_LIST:
            print("[Skip]", op_name, ": Blacklisted", file=sys.stderr)
            return True
        for prefix in PREFIX_BLACK_LIST:
            if op_name.startswith(prefix):
                print("[Skip]", op_name, ": Blacklisted", file=sys.stderr)
                return True
        return False

    result = []
    for op_name in list_op():
        op_name = op_name.value
        if is_black_listed(op_name):
            continue
        op: relay.Op = get_op(op_name)
        assert op.name == op_name
        attrs = op.attrs_type_key
        fcompute = op.get_attr("FTVMCompute")
        fschedule = op.get_attr("FTVMSchedule")
        pattern = op.get_attr("TOpPattern")
        skip_reasons = []
        if not fcompute:
            skip_reasons.append("No-FTVMCompute")
        if not fschedule:
            skip_reasons.append("No-FTVMSchedule")
        if pattern is None:
            skip_reasons.append("No-TOpPattern")
        if skip_reasons:
            print("[Skip]", op_name, ":", ", ".join(skip_reasons), file=sys.stderr)
            continue
        if not attrs:
            attrs = ""
        pattern = pattern_map[pattern]
        result.append((op_name, attrs, pattern))
    return result


def main():
    ops = collect_op()
    print("OP_MAP = {")
    for op_name, attrs, pattern in ops:
        if op_name in MNM_OP_NAME:
            mnm_op_name = MNM_OP_NAME[op_name]
        else:
            mnm_op_name = "mnm.op." + op_name
        print(f'    "{mnm_op_name}": ["{op_name}", "{attrs}", "{pattern}"],')
    print("}")


if __name__ == "__main__":
    main()
